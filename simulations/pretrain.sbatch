#!/bin/bash
# =============================================================================
# Quest (SLURM) 提交脚本：denoiser 预训练（= notebook cell-4 / 论文严格配置）
# 产物: checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth
#       （即 train_and_sample.ipynb cell-4 要加载的那个 checkpoint）
# 用法: 在 simulations/ 目录下  ->  sbatch pretrain.sbatch
# =============================================================================

# ---- 资源申请：★ 三处需要你按自己的 Quest 账号/环境修改（标了 ★）----
#SBATCH --account=pXXXXXX           # ★ 你的 Quest allocation (项目号, 形如 p12345)
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

# ---- 激活环境：★ 按你在 Quest 上的实际方式修改 ----
module purge
module load python-anaconda3        # ★ 看 Quest 用哪个 (python-anaconda3 / mamba / ...)
eval "$(conda shell.bash hook)"
conda activate ET_cuda              # ★ 你在 Quest 上建好的 conda 环境名

# ---- 运行（参数 = notebook cell-4 的论文严格配置）----
# main.py 会自动 chdir 到自身所在目录，所以从哪提交都行；但建议就在 simulations/ 下提交。
cd "$SLURM_SUBMIT_DIR"

python main.py \
    --x_type linear_latent --pretrain \
    --N_0 65536 --pre_num_episodes 20 --pre_bs 32 --pre_lr 1e-4 --T 200 \
    --d_outer 64 --d_inner 16 --r_off 9 \
    --pretrain_ckpt_folder checkpoints/pretrained

echo "DONE -> $SLURM_SUBMIT_DIR/checkpoints/pretrained/linear_latent_pretrain_epoch_20.pth"
