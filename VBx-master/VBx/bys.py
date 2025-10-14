#!/usr/bin/env python

import argparse
import os
import itertools
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.metrics import silhouette_score

import fastcluster
import h5py
import kaldi_io
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
import json
from datetime import datetime

from diarization_lib import read_xvector_timing_dict, l2_norm, \
    cos_similarity, twoGMMcalib_lin, merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VBx import VBx


def write_output(fp, out_labels, starts, ends, file_name):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


def run_vbx_with_params(x, plda_mu, plda_tr, plda_psi, lda_dim, 
                        Fa, Fb, loopP, init_smoothing, ahc_labels):
    """运行VBx算法并返回聚类标签和特征"""
    qinit = np.zeros((len(ahc_labels), np.max(ahc_labels) + 1))
    qinit[range(len(ahc_labels)), ahc_labels] = 1.0
    qinit = softmax(qinit * init_smoothing, axis=1)
    
    fea = (x - plda_mu).dot(plda_tr.T)[:, :lda_dim]
    q, sp, L = VBx(
        fea, plda_psi[:lda_dim],
        pi=qinit.shape[1], gamma=qinit,
        maxIters=40, epsilon=1e-6,
        loopProb=loopP, Fa=Fa, Fb=Fb)
    
    return np.argsort(-q, axis=1)[:, 0], fea


def compute_silhouette_score(labels, features):
    """计算轮廓系数"""
    if len(np.unique(labels)) < 2 or len(labels) <= 2:
        return -1.0  # 无效聚类
    
    try:
        return silhouette_score(features, labels)
    except:
        return -1.0


def process_recording_for_optimization(segs_dict, arkit, plda_mu, plda_tr, plda_psi, 
                                     lda_dim, Fa, Fb, loopP, init_smoothing, threshold, 
                                     xvec_transform):
    """处理录音并返回聚类质量"""
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
    all_labels = []
    all_features = []
    
    for file_name, segs in recit:
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        # AHC聚类
        scr_mx = cos_similarity(x)
        thr, _ = twoGMMcalib_lin(scr_mx.ravel())
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(
            scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        ahc_labels = fcluster(lin_mat, -(thr + threshold) + adjust,
                            criterion='distance') - 1

        # VBx聚类
        vb_labels, features = run_vbx_with_params(x, plda_mu, plda_tr, plda_psi, 
                                                lda_dim, Fa, Fb, loopP, init_smoothing, ahc_labels)

        all_labels.extend(vb_labels)
        all_features.extend(features.tolist())
    
    return np.array(all_labels), np.array(all_features)


class BayesianOptimizationVisualizer:
    """贝叶斯优化可视化类"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []
        self.best_score = -float('inf')
        self.best_params = None
        mkdir_p(output_dir)
        
    def update(self, iteration, params, score):
        """更新历史记录"""
        record = {
            'iteration': iteration,
            'params': params,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(record)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            
    def create_visualizations(self):
        """创建所有可视化图表"""
        self._plot_convergence()
        self._plot_parameter_evolution()
        self._plot_parameter_relationships()
        self._plot_3d_parameter_space()
        self._save_results_summary()
        
    def _plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 6))
        
        # 最佳分数随迭代的变化
        best_scores = [max([h['score'] for h in self.history[:i+1]]) 
                      for i in range(len(self.history))]
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.history) + 1), [h['score'] for h in self.history], 
                'o-', alpha=0.7, label='每次迭代的分数')
        plt.plot(range(1, len(best_scores) + 1), best_scores, 
                'r-', linewidth=2, label='最佳分数')
        plt.xlabel('迭代次数')
        plt.ylabel('轮廓系数')
        plt.title('优化收敛曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 分数分布直方图
        plt.subplot(1, 2, 2)
        scores = [h['score'] for h in self.history]
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(self.best_score, color='red', linestyle='--', 
                   label=f'最佳分数: {self.best_score:.4f}')
        plt.xlabel('轮廓系数')
        plt.ylabel('频次')
        plt.title('分数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'convergence_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_parameter_evolution(self):
        """绘制参数演化图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        parameters = ['Fa', 'Fb', 'loopP']
        
        for i, param in enumerate(parameters):
            values = [h['params'][param] for h in self.history]
            scores = [h['score'] for h in self.history]
            
            ax = axes[i]
            scatter = ax.scatter(range(1, len(values) + 1), values, c=scores, 
                               cmap='viridis', s=50, alpha=0.7)
            ax.set_xlabel('epochs')
            ax.set_ylabel(param)
            ax.set_title(f'{param} Param change')
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Silhouette coefficient')
        
        # 第四个图：参数范围
        ax = axes[3]
        for param in parameters:
            values = [h['params'][param] for h in self.history]
            ax.plot(range(1, len(values) + 1), values, 'o-', alpha=0.7, label=param)
        
        ax.set_xlabel('epochs')
        ax.set_ylabel('Param value')
        ax.set_title('Param change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_parameter_relationships(self):
        """绘制参数关系图"""
        df = pd.DataFrame([
            {**h['params'], 'score': h['score']} 
            for h in self.history
        ])
        
        # 散点图矩阵
        g = sns.PairGrid(df, vars=['Fa', 'Fb', 'loopP', 'score'], 
                        diag_sharey=False, height=2.5)
        g.map_lower(sns.scatterplot, s=50, alpha=0.7)
        g.map_diag(sns.histplot, kde=True)
        g.map_upper(sns.kdeplot, fill=True, alpha=0.5)
        
        plt.suptitle('Parameter relationship matrix', y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'parameter_relationships.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_3d_parameter_space(self):
        """绘制3D参数空间图"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        Fa = [h['params']['Fa'] for h in self.history]
        Fb = [h['params']['Fb'] for h in self.history]
        loopP = [h['params']['loopP'] for h in self.history]
        scores = [h['score'] for h in self.history]
        
        scatter = ax.scatter(Fa, Fb, loopP, c=scores, cmap='viridis', 
                           s=100, alpha=0.8, depthshade=True)
        
        ax.set_xlabel('Fa')
        ax.set_ylabel('Fb')
        ax.set_zlabel('loopP')
        ax.set_title('3D参数空间探索')
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('轮廓系数')
        
        plt.savefig(os.path.join(self.output_dir, '3d_parameter_space.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_results_summary(self):
        """保存结果摘要"""
        summary = {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'total_iterations': len(self.history),
            'average_score': np.mean([h['score'] for h in self.history]),
            'std_score': np.std([h['score'] for h in self.history]),
            'optimization_history': self.history
        }
        
        with open(os.path.join(self.output_dir, 'optimization_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式的结果
        df = pd.DataFrame([
            {**h['params'], 'score': h['score'], 'iteration': h['iteration']} 
            for h in self.history
        ])
        df.to_csv(os.path.join(self.output_dir, 'optimization_results.csv'), index=False)


def evaluate_parameters_silhouette(Fa, Fb, loopP, args, visualizer=None, iteration=0):
    """使用轮廓系数评估参数"""
    Fa = max(0.1, min(Fa, 20.0))
    Fb = max(0.1, min(Fb, 20.0))
    loopP = max(0.0, min(loopP, 1.0))
    
    print(f"迭代 {iteration}: 测试参数 Fa={Fa:.3f}, Fb={Fb:.3f}, loopP={loopP:.3f}")
    
    # 处理所有录音文件
    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    
    # 重置文件读取器
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    segs_dict = read_xvector_timing_dict(args.segments_file)
    
    # 处理录音并获取聚类结果
    labels, features = process_recording_for_optimization(
        segs_dict, arkit, plda_mu, plda_tr, plda_psi,
        args.lda_dim, Fa, Fb, loopP, args.init_smoothing,
        args.threshold, args.xvec_transform
    )
    
    # 计算轮廓系数
    silhouette = compute_silhouette_score(labels, features)
    print(f"迭代 {iteration}: 轮廓系数 = {silhouette:.4f}")
    
    # 更新可视化器
    if visualizer is not None:
        params = {'Fa': Fa, 'Fb': Fb, 'loopP': loopP}
        visualizer.update(iteration, params, silhouette)
    
    return silhouette


def process_recording_normal(segs_dict, arkit, plda_mu, plda_tr, plda_psi, 
                           lda_dim, Fa, Fb, loopP, init_smoothing, threshold, 
                           xvec_transform):
    """正常处理录音并输出结果"""
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
    
    for file_name, segs in recit:
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        # AHC聚类
        scr_mx = cos_similarity(x)
        thr, _ = twoGMMcalib_lin(scr_mx.ravel())
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(
            scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        ahc_labels = fcluster(lin_mat, -(thr + threshold) + adjust,
                            criterion='distance') - 1

        # VBx聚类
        vb_labels, _ = run_vbx_with_params(x, plda_mu, plda_tr, plda_psi, 
                                         lda_dim, Fa, Fb, loopP, init_smoothing, ahc_labels)

        # 写入输出文件
        starts = []
        ends = []
        for seg_name in seg_names:
            start, end = segs_dict[seg_name]
            starts.append(start)
            ends.append(end)
        
        
        print(f"处理完成 {file_name}, 发现 {len(np.unique(vb_labels))} 个说话人")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str,
                        choices=['normal', 'optimize-silhouette'],
                        help='运行模式: normal为正常聚类，optimize-silhouette为基于轮廓系数的无监督优化')
    parser.add_argument('--xvec-ark-file', required=True, type=str,
                        help='x-vector ark文件')
    parser.add_argument('--segments-file', required=True, type=str,
                        help='分段文件')
    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='x-vector变换文件')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='PLDA模型文件')
    parser.add_argument('--threshold', required=True, type=float,
                        help='AHC阈值')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='LDA维度')
    parser.add_argument('--Fa', required=False, type=float, default=1.0,
                        help='VBx Fa参数')
    parser.add_argument('--Fb', required=False, type=float, default=1.0,
                        help='VBx Fb参数')
    parser.add_argument('--loopP', required=False, type=float, default=0.5,
                        help='VBx loopP参数')
    parser.add_argument('--init-smoothing', required=False, type=float,
                        default=5.0, help='初始化平滑参数')
    parser.add_argument('--optimization-iterations', required=False, type=int,
                        default=20, help='优化迭代次数')
    parser.add_argument('--optimization-log', required=False, type=str,
                        default='optimization_log.json',
                        help='优化日志文件')
    parser.add_argument('--visualization-dir', required=False, type=str,
                        default='optimization_visualization',
                        help='可视化结果输出目录')
    
    args = parser.parse_args()
    
    
    if args.mode == 'normal':
        # 正常聚类模式
        print(f"使用参数进行正常聚类: Fa={args.Fa}, Fb={args.Fb}, loopP={args.loopP}")
        
        kaldi_plda = read_plda(args.plda_file)
        plda_mu, plda_tr, plda_psi = kaldi_plda
        W = np.linalg.inv(plda_tr.T.dot(plda_tr))
        B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
        acvar, wccn = eigh(B, W)
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]
        
        arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
        segs_dict = read_xvector_timing_dict(args.segments_file)
        
        process_recording_normal(
            segs_dict, arkit, plda_mu, plda_tr, plda_psi,
            args.lda_dim, args.Fa, args.Fb, args.loopP, args.init_smoothing,
            args.threshold, args.xvec_transform
        )
        
        print("正常聚类完成!")
        
    elif args.mode == 'optimize-silhouette':
        # 基于轮廓系数的无监督参数优化模式
        print("开始基于轮廓系数的贝叶斯优化...")
        
        # 创建可视化器
        visualizer = BayesianOptimizationVisualizer(args.visualization_dir)
        
        pbounds = {
            'Fa': (0.1, 1.0),
            'Fb': (3.0, 8.0),
            'loopP': (0.8, 1.0)
        }
        
        # 包装评估函数以包含迭代信息
        def evaluation_function(Fa, Fb, loopP):
            nonlocal iteration_count
            score = evaluate_parameters_silhouette(
                Fa, Fb, loopP, args, visualizer, iteration_count)
            iteration_count += 1
            return score
        
        iteration_count = 1
        optimizer = BayesianOptimization(
            f=evaluation_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2,
            allow_duplicate_points=True  # 添加这行
        )
        
        if os.path.exists(args.optimization_log):
            try:
                load_logs(optimizer, logs=[args.optimization_log])
                print(f"已加载优化日志: {args.optimization_log}")
            except Exception as e:
                print(f"无法加载优化日志: {e}, 将重新开始")
        
        logger = JSONLogger(path=args.optimization_log)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        
        optimizer.maximize(
            init_points=5,
            n_iter=args.optimization_iterations,
        )
        
        # 创建可视化图表
        print("创建优化过程可视化...")
        visualizer.create_visualizations()
        
        print("基于轮廓系数的优化完成!")
        print(f"最佳参数: {optimizer.max['params']}")
        print(f"最佳轮廓系数: {optimizer.max['target']:.4f}")
        print(f"可视化结果已保存到: {args.visualization_dir}")
        
        # 使用最佳参数运行最终聚类
        best_params = optimizer.max['params']
        print(f"使用最佳参数进行最终聚类: Fa={best_params['Fa']:.3f}, Fb={best_params['Fb']:.3f}, loopP={best_params['loopP']:.3f}")
        
        kaldi_plda = read_plda(args.plda_file)
        plda_mu, plda_tr, plda_psi = kaldi_plda
        W = np.linalg.inv(plda_tr.T.dot(plda_tr))
        B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
        acvar, wccn = eigh(B, W)
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]
        
        arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
        segs_dict = read_xvector_timing_dict(args.segments_file)
        
        process_recording_normal(
            segs_dict, arkit, plda_mu, plda_tr, plda_psi,
            args.lda_dim, best_params['Fa'], best_params['Fb'], best_params['loopP'], 
            args.init_smoothing, args.threshold, args.xvec_transform
        )
        
        print("最终聚类完成!")
        
    else:
        print("请选择正确的运行模式")


if __name__ == '__main__':
    main()