# Basic INRs
python eval_ginr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/bunny_v1       --n_fourier 100 --mesh data_generation/bunny/reconstruction/bun_zipper.ply --key bunny
python eval_ginr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/protein_1AA7_A --n_fourier 100 --mesh data_generation/proteins/obj_files/1AA7_A.obj       --key protein_1AA7_A

# Super-resolution
python eval_super_resolution.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/bunny_v1 --n_fourier 7

# Reaction-diffusion
python eval_reaction_diffusion.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/bunny_time_full --n_fourier 100 --time=True --gif_sample_every 5

# Weather modelling
python eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_gustsfc --n_fourier 100 --time=True --time_factor=2 --cmap Spectral --append gustsfc
python eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_dpt2m   --n_fourier 100 --time=True --time_factor=2 --cmap hot      --append dpt2m
python eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_tcdcclm --n_fourier 100 --time=True --time_factor=2 --cmap Blues    --append tcdcclm

# Weather modelling + super-resolution (see comments in commands_training.sh)
python eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_gustsfc_sr_cut --n_fourier 34 --time=True --time_factor=2 --cmap Spectral --append gustsfc
python eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_dpt2m_sr_cut   --n_fourier 34 --time=True --time_factor=2 --cmap hot      --append dpt2m
python eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_tcdcclm_sr_cut --n_fourier 34 --time=True --time_factor=2 --cmap Blues    --append tcdcclm

