# -*- coding: utf-8 -*-
import csv
import json
import os
import time
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set
import requests


class GitHubPRFetcher:
    def __init__(self, token, output_dir='pr_data', max_workers=5, max_prs=5000):
        """初始化PR数据获取器（仅负责数据获取与特征提取）"""
        self.token = token
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.output_dir = output_dir
        self.max_prs = max_prs  # 每个仓库最多获取5000个PR
        self.max_workers = max_workers  # 线程数
        os.makedirs(self.output_dir, exist_ok=True)

        # 线程安全控制
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.api_calls = 0
        self.rate_limit_reset = time.time() + 3600
        self.lock = threading.Lock()

    def _handle_rate_limit(self):
        """线程安全的GitHub API限流处理"""
        with self.lock:
            self.api_calls += 1
            if self.api_calls % 60 == 0:
                try:
                    resp = requests.get("https://api.github.com/rate_limit", headers=self.headers)
                    resp.raise_for_status()
                    data = resp.json()
                    remaining = data['resources']['core']['remaining']
                    self.rate_limit_reset = data['resources']['core']['reset']

                    if remaining < 10:
                        sleep_time = self.rate_limit_reset - time.time() + 10
                        print(f"API限流，等待{sleep_time:.2f}秒...")
                        time.sleep(sleep_time)
                        self.api_calls = 0
                except Exception as e:
                    print(f"检查限流失败: {e}")
                    time.sleep(30)
        time.sleep(0.5)

    def fetch_pr_basic_info(self, owner: str, repo: str) -> List[Dict]:
        """获取PR基础信息（仅保留有创建/关闭时间的PR，用于计算TTC）"""
        print(f"[{owner}/{repo}] 获取PR基础信息...")
        pr_list = []
        page = 1
        per_page = 100

        while len(pr_list) < self.max_prs:
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            params = {
                "state": "closed",
                "per_page": per_page,
                "page": page,
                "sort": "created",
                "direction": "asc"
            }

            try:
                resp = requests.get(url, headers=self.headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break  # 无数据表示已获取所有页面，正常终止

                # 过滤：仅保留有创建和关闭时间的PR（否则无法计算TTC）
                valid_prs = [pr for pr in data if pr['created_at'] and pr['closed_at']]
                if len(pr_list) + len(valid_prs) > self.max_prs:
                    valid_prs = valid_prs[:self.max_prs - len(pr_list)]

                pr_list.extend(valid_prs)
                print(f"[{owner}/{repo}] 已获取有效PR: {len(pr_list)}/{self.max_prs}")
                page += 1
                self._handle_rate_limit()
            except Exception as e:
                print(f"[{owner}/{repo}] 获取第{page}页失败: {e}")
                # 核心修改：跳过当前失败页面，继续尝试下一页（原逻辑为break）
                page += 1
                # 失败后短暂休眠，避免频繁请求触发更多错误
                time.sleep(2)

        return pr_list

    def fetch_contributors(self, owner: str, repo: str) -> Set[str]:
        """获取仓库核心贡献者（用于判断is_core_member特征）"""
        contributors = set()
        page = 1
        while True:
            try:
                resp = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contributors",
                    headers=self.headers,
                    params={"per_page": 100, "page": page}
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                contributors.update([c['login'] for c in data])
                page += 1
                self._handle_rate_limit()
            except Exception as e:
                print(f"[{owner}/{repo}] 获取贡献者失败: {e}")
                break
        print(f"[{owner}/{repo}] 核心贡献者数量: {len(contributors)}")
        return contributors

    def fetch_author_prev_prs(self, owner: str, repo: str, author_login: str, pr_created_at: str) -> int:
        """统计作者在当前PR创建前的历史PR数量（prev_PRs特征）"""
        count = 0
        page = 1
        while True:
            try:
                resp = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/pulls",
                    headers=self.headers,
                    params={
                        "state": "all",
                        "author": author_login,
                        "per_page": 100,
                        "page": page
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break

                # 仅统计创建时间早于当前PR的历史PR
                prev_prs = [pr for pr in data if pr['created_at'] and pr['created_at'] < pr_created_at]
                count += len(prev_prs)

                # 若当前页最后一个PR仍早于目标时间，继续翻页
                last_pr_time = data[-1]['created_at'] if data else None
                if last_pr_time and last_pr_time < pr_created_at:
                    page += 1
                    self._handle_rate_limit()
                else:
                    break
            except Exception as e:
                print(f"[{owner}/{repo}] 统计{author_login}历史PR失败: {e}")
                break
        return count

    def _extract_single_pr_features(self, owner: str, repo: str, pr: Dict, contributors: Set[str]) -> Dict:
        """提取单个PR的完整特征（含TTC标签）"""
        features = {}
        pr_number = pr['number']
        pr_created_at = pr['created_at']
        author_login = pr['user']['login']
        text = (pr['title'] or "") + " " + (pr['body'] or "")
        text_lower = text.lower()

        # -------------------------- 1-7. 基础与文本二元特征 --------------------------
        features['assignees'] = len(pr['assignees']) if pr['assignees'] else 0
        keywords = ['test', 'bug', 'feature', 'improve', 'document', 'refactor']
        for kw in keywords:
            features[f'has_{kw}'] = 1 if kw in text_lower else 0

        # -------------------------- 8-10. 目录与文件结构特征 --------------------------
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files",
                headers=self.headers
            )
            resp.raise_for_status()
            files = resp.json()
            self._handle_rate_limit()
        except Exception as e:
            print(f"[{owner}/{repo}] PR #{pr_number} 文件获取失败: {e}")
            files = []

        # 8. directories: 修改的目录数（去重）
        directories = set(os.path.dirname(f['filename']) for f in files if '/' in f['filename'])
        features['directories'] = len(directories)

        # 9. language_types: 编程语言类型数
        lang_ext = {'py': 'Python', 'java': 'Java', 'c': 'C', 'cpp': 'C++', 'js': 'JS', 'ts': 'TS', 'go': 'Go'}
        languages = set(lang_ext[f['filename'].split('.')[-1].lower()] for f in files
                        if '.' in f['filename'] and f['filename'].split('.')[-1].lower() in lang_ext)
        features['language_types'] = len(languages)

        # 10. file_types: 文件类型数（按后缀）
        file_types = set(f['filename'].split('.')[-1].lower() for f in files if '.' in f['filename'])
        features['file_types'] = len(file_types)

        # -------------------------- 11-18. 修改规模特征 --------------------------
        lines_added = sum(f.get('additions', 0) for f in files)
        lines_deleted = sum(f.get('deletions', 0) for f in files)
        features['lines_added'] = lines_added
        features['lines_deleted'] = lines_deleted

        # 13-15. 代码段统计（按patch中的+/-行）
        segs_added = sum(1 for f in files if 'patch' in f
                         for line in f['patch'].split('\n') if line.startswith('+') and not line.startswith('@@'))
        segs_deleted = sum(1 for f in files if 'patch' in f
                           for line in f['patch'].split('\n') if line.startswith('-') and not line.startswith('@@'))
        segs_changed = sum(1 for f in files if 'patch' in f
                           if any(line.startswith('+') for line in f['patch'].split('\n') if not line.startswith('@@'))
                           and any(
            line.startswith('-') for line in f['patch'].split('\n') if not line.startswith('@@')))
        features['segs_added'] = segs_added
        features['segs_deleted'] = segs_deleted
        features['segs_changed'] = segs_changed

        # 16-18. 文件操作统计
        features['files_added'] = sum(1 for f in files if f.get('status') == 'added')
        features['files_deleted'] = sum(1 for f in files if f.get('status') == 'removed')
        features['files_changed'] = sum(1 for f in files if f.get('status') == 'modified')

        # -------------------------- 19-24. 作者相关特征 --------------------------
        features['file_developer'] = 1  # 简化处理：默认1个开发者（复杂场景可扩展）
        # 20. change_num: 当前PR的commit数
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits",
                headers=self.headers
            )
            resp.raise_for_status()
            commits = resp.json()
            self._handle_rate_limit()
            features['change_num'] = len(commits) if commits else 0
        except Exception as e:
            print(f"[{owner}/{repo}] PR #{pr_number} commits获取失败: {e}")
            features['change_num'] = 0
        features['files_modified'] = 0  # 简化处理：文件历史修改次数（可扩展）
        features['is_core_member'] = 1 if author_login in contributors else 0
        features['commits'] = features['change_num']  # 复用commit数
        features['prev_PRs'] = self.fetch_author_prev_prs(owner, repo, author_login, pr_created_at)

        # -------------------------- 25-26. 文本长度特征 --------------------------
        features['title_words'] = len(pr['title'].split()) if pr['title'] else 0
        features['body_words'] = len(pr['body'].split()) if pr['body'] else 0

        # -------------------------- TTC标签（任务一目标） --------------------------
        created = datetime.fromisoformat(pr_created_at.replace('Z', '+00:00'))
        closed = datetime.fromisoformat(pr['closed_at'].replace('Z', '+00:00'))
        features['TTC'] = round((closed - created).total_seconds() / 3600, 2)  # 单位：小时

        # 补充基础信息（用于时间切分和仓库标记）
        features['pr_number'] = pr_number
        features['created_at'] = pr_created_at
        features['repo'] = f"{owner}/{repo}"

        return features

    def process_repo(self, owner: str, repo: str) -> pd.DataFrame:
        """处理单个仓库，生成特征DataFrame"""
        # 1. 获取基础数据
        pr_basic_list = self.fetch_pr_basic_info(owner, repo)
        if not pr_basic_list:
            print(f"[{owner}/{repo}] 无有效PR，跳过")
            return pd.DataFrame()
        contributors = self.fetch_contributors(owner, repo)

        # 2. 多线程提取特征
        print(f"[{owner}/{repo}] 多线程提取{len(pr_basic_list)}个PR的特征...")
        features_list = []
        futures = []
        for pr in pr_basic_list:
            future = self.executor.submit(
                self._extract_single_pr_features,
                owner, repo, pr, contributors
            )
            futures.append(future)

        # 收集结果
        for i, future in enumerate(as_completed(futures)):
            feature = future.result()
            if feature:
                features_list.append(feature)
            if (i + 1) % 50 == 0:
                print(f"[{owner}/{repo}] 已处理 {i + 1}/{len(pr_basic_list)} 个PR")

        # 3. 转换为DataFrame并保存
        if not features_list:
            return pd.DataFrame()
        df = pd.DataFrame(features_list)
        save_path = f"{self.output_dir}/{owner}_{repo}_ttc_features.csv"
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"[{owner}/{repo}] 特征数据已保存至: {save_path}")
        print(f"[{owner}/{repo}] 最终有效样本数: {len(df)}")

        return df

    def __del__(self):
        """关闭线程池"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()


# -------------------------- 主函数：运行数据获取 --------------------------
if __name__ == "__main__":
    # 配置参数（替换为你的GitHub Token！）
    # GITHUB_TOKEN = "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"  # 重要：勿泄露，使用后重置
    GITHUB_TOKEN = "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"
    # REPOS = [("tensorflow", "tensorflow")]  # 至少2个仓库（Level 2要求）
    REPOS = [("pytorch", "pytorch")]  # 至少2个仓库（Level 2要求）
    MAX_PRS = 5000  # 每个仓库最多5000个PR
    MAX_WORKERS = 8  # 线程数（5-10为宜）

    # 初始化获取器并运行
    fetcher = GitHubPRFetcher(
        token=GITHUB_TOKEN,
        max_workers=MAX_WORKERS,
        max_prs=MAX_PRS
    )

    # 处理所有仓库并合并数据
    all_dfs = []
    for owner, repo in REPOS:
        print(f"\n===== 开始处理仓库: {owner}/{repo} =====")
        repo_df = fetcher.process_repo(owner, repo)
        if not repo_df.empty:
            all_dfs.append(repo_df)
        print(f"===== 完成处理仓库: {owner}/{repo} =====\n")

    # 合并所有仓库数据（可选，方便后续建模）
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = "pr_data/combined_ttc_features.csv"
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        print(f"所有仓库合并数据已保存至: {combined_path}")
        print(f"合并后总样本数: {len(combined_df)}")

    print("\n数据获取脚本运行完成！生成的CSV文件可用于后续建模。")