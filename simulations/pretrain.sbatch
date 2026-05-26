#!/bin/bash
# =============================================================================
# Quest (SLURM) 提交脚本：denoiser 预训练（= notebook cell-4 / 论文严格配置）
# 产物: checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth
#       （即 train_and_sample.ipynb cell-4 要加载的那个 checkpoint）
# 用法: 在 simulations/ 目录下  ->  sbatch pretrain.sbatch
# =============================================================================

# ---- 资源申请：★ 三处需要你按自己的 Quest 账号/环境修改（标了 ★）----
#SBATCH --account=p32889            # Quest allocation (项目号)
#SBATCH --partition=gengpu          # Quest GPU 分区（gengpu 有 A100）
#SBATCH --gres=gpu:a100:1           # 1 张 A100（若该分区无 a100，可去掉:a100 写 gpu:1）
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00             # A100 上几分钟即可，给 1h 余量
#SBATCH --job-name=ggdm_pretrain
#SBATCH --output=pretrain_%j.out    # stdout（print 信息）
#SBATCH --error=pretrain_%j.err     # stderr（tqdm 进度条会进这里）

set -euo pipefail

# ---- 环境：照搬同账号 acoustic 项目里已验证可在 gengpu A100 上跑通的配方 ----
# dps_uq 装在 project space（home 配额满），不在 conda 的按名搜索路径上；用绝对路径
# 调用解释器比 `conda activate` 在 SLURM 批处理里更稳。不要 `module purge`：之前那版
# purge + conda activate 触发了 cuDNN error: CUDNN_STATUS_NOT_INITIALIZED。
export PYTHONNOUSERSITE=1            # 忽略 ~/.local，避免任何包覆盖 env
PY=/projects/p32889/.conda/envs/dps_uq/bin/python

# main.py 会自动 chdir 到自身所在目录，所以从哪提交都行；但建议就在 simulations/ 下提交。
cd "$SLURM_SUBMIT_DIR"

# ---- 诊断（若再失败，这几行能告诉我们 GPU/torch 是否正常）----
echo "host: $(hostname)"
nvidia-smi || true
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"

# ---- 运行：只做 notebook cell-4 的训练（--pretrain_only 跳过 main.py 后续的无条件采样）----
"$PY" main.py \
    --x_type linear_latent --pretrain --pretrain_only \
    --N_0 65536 --pre_num_episodes 20 --pre_bs 32 --pre_lr 1e-4 --T 200 \
    --d_outer 64 --d_inner 16 --r_off 9 \
    --pretrain_ckpt_folder checkpoints/pretrained

echo "DONE -> $SLURM_SUBMIT_DIR/checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth"
