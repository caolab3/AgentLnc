#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict

class LocalDatabaseHandler:
    """本地文件数据库处理器，替代MySQL查询"""
    
    def __init__(self):
        # 获取当前脚本路径，并加上 /database
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "database")
        
        # 检查database目录是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Database directory not found: {self.data_dir}\n"
                f"Please ensure the 'database' folder exists in: {script_dir}"
            )
        
        self.cache = {}
        self.indices = {}
        
        # 自动检测和报告可用的数据文件
        self._check_available_files()
    
    def _check_available_files(self):
        """检查并报告可用的数据文件"""
        expected_files = ['NPInter.txt', 'RNAInter.txt', 'RNAInter_mRNA.txt', 'starBase.txt','GTEx_Tissue.txt', 'BIOGRID_HINT_Merge_PPI.txt']
        available_files = []
        missing_files = []
        
        for filename in expected_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                available_files.append(filename)
            else:
                missing_files.append(filename)
        
        print(f"[INFO] Database directory: {self.data_dir}")
        if available_files:
            print(f"[INFO] Available database files: {', '.join(available_files)}")
        if missing_files:
            print(f"[WARNING] Missing database files: {', '.join(missing_files)}")
    
    def load_tsv_file(self, filename: str, delimiter="\t") -> Tuple[List[str], List[List[str]]]:
        """加载TSV文件并返回列名和数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        # 使用缓存避免重复读取
        if filename in self.cache:
            return self.cache[filename]
        
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found: {filepath}")
            return [], []
        
        columns = []
        data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    # 第一行作为列名
                    columns = lines[0].rstrip('\n\r').split(delimiter)
                    # 后续行为数据
                    for line_num, line in enumerate(lines[1:], start=2):
                        row = line.rstrip('\n\r').split(delimiter)
                        if len(row) == len(columns):  # 确保列数匹配
                            data.append(row)
                        elif line.strip():  # 忽略空行
                            print(f"[WARNING] Line {line_num} in {filename} has {len(row)} columns, expected {len(columns)}")
                
                print(f"[INFO] Loaded {filename}: {len(data)} rows, {len(columns)} columns")
                
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
            return [], []
        
        self.cache[filename] = (columns, data)
        return columns, data
    
    def build_index(self, filename: str, index_columns: List[str]):
        """为指定列构建索引以加速搜索"""
        columns, data = self.load_tsv_file(filename)
        
        if not columns or not data:
            return
        
        if filename not in self.indices:
            self.indices[filename] = {}
        
        for col_name in index_columns:
            if col_name not in columns:
                print(f"[WARNING] Column '{col_name}' not found in {filename}")
                continue
                
            col_idx = columns.index(col_name)
            index = defaultdict(list)
            
            for row_idx, row in enumerate(data):
                if col_idx < len(row):
                    value = row[col_idx].lower()
                    # 建立倒排索引
                    words = re.findall(r'\w+', value)
                    for word in words:
                        index[word].append(row_idx)
            
            self.indices[filename][col_name] = index
            print(f"[INFO] Built index for {filename}.{col_name}")
    
    def search_database(self, filename: str, search_conditions: List[Tuple[str, str]]) -> Tuple[List[str], List[List[str]]]:
        """
        模拟SQL查询
        search_conditions: [(column_name, search_pattern), ...]
        """
        columns, data = self.load_tsv_file(filename)
        
        if not columns or not data:
            return [], []
        
        results = []
        
        for row in data:
            match = False
            for col_name, pattern in search_conditions:
                if col_name not in columns:
                    continue
                    
                col_idx = columns.index(col_name)
                if col_idx < len(row):
                    cell_value = row[col_idx].lower()
                    search_term = pattern.replace('%', '').lower()
                    
                    # 模拟SQL LIKE操作
                    if search_term in cell_value:
                        match = True
                        break
            
            if match:
                results.append(row)
        
        return columns, results

# 全局数据库处理器实例
db_handler = None

def init_local_db():
    """初始化本地数据库处理器 - 无需参数，自动使用相对路径"""
    global db_handler
    db_handler = LocalDatabaseHandler()

    db_files = {
        'NPInter': 'NPInter.txt',
        'RNAInter': 'RNAInter.txt',
        'RNAInter_mRNA': 'RNAInter_mRNA.txt',
        'starBase': 'starBase.txt',
        'GTEx_Tissue': 'GTEx_Tissue.txt',
        'BIOGRID_HINT_Merge_PPI': 'BIOGRID_HINT_Merge_PPI.txt', 
    }
    
   
    
    for db_name, filename in db_files.items():
        filepath = os.path.join(db_handler.data_dir, filename)
        if os.path.exists(filepath):
            print(f"[INFO] Indexing {db_name}...")
            if db_name in ['NPInter', 'RNAInter', 'starBase']:
                db_handler.build_index(filename, ['LncRNA Name', 'RBP Name'])
            elif db_name == 'GTEx_Tissue':
                db_handler.build_index(filename, ['Description', 'Name'])
            elif db_name == 'BIOGRID_HINT_Merge_PPI':
                db_handler.build_index(filename, ['Gene Name A', 'Gene Name B'])
            elif db_name == 'RNAInter_mRNA':
                db_handler.build_index(filename, ['mRNA Name', 'RBP Name'])

def query_database_local(gene: str, table_name: str, search_conditions: List[Tuple[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    """
    替代原来的query_database函数
    """
    if db_handler is None:
        # 自动初始化
        init_local_db()
    
    # 映射表名到文件名
    table_to_file = {
        'NPInter': 'NPInter.txt',
        'RNAInter': 'RNAInter.txt',
        'RNAInter_mRNA': 'RNAInter_mRNA.txt',
        'starBase': 'starBase.txt',
        'GTEx_Tissue': 'GTEx_Tissue.txt',
        'BIOGRID_HINT_Merge_PPI': 'BIOGRID_HINT_Merge_PPI.txt',
    }
    
    filename = table_to_file.get(table_name)
    if not filename:
        print(f"[WARNING] Unknown table: {table_name}")
        return [], []
    
    # 转换搜索条件格式
    conditions = []
    for col, pattern_func in search_conditions:
        pattern = pattern_func(gene)
        conditions.append((col, pattern))
    
    return db_handler.search_database(filename, conditions)