import os
from ae_split_demo import main, plot_latent
os.chdir('/home/sgrinek/PycharmProjects/DCAE')
art = main([
    '--m','10','--ell', '7', '--N','6000','--epochs-mse','10000','--epochs-dcae',
      '10000','--lambda-dcae','1e-4',
      '--batch-size','4096','--threads','56','--interop-threads','8',
      '--num-workers','8','--prefetch-factor','4','--compile', '--early-stop',
      "--patience", "100", "--min-delta", '1e-7'])

X_np      = art['X_input']              # (N, m)
Z_mse     = art['mse']['Z']             # (N, k)
Xhat_mse  = art['mse']['Xhat']          # (N, m)
Z_final   = art['final']['Z']           # (N, k)
Xhat_final= art['final']['Xhat']        # (N, m)

enc_mse_fn   = art['mse']['encoder_fn']
dec_mse_fn   = art['mse']['decoder_fn']
ae_mse_fn    = art['mse']['autoencoder_fn']
enc_final_fn = art['final']['encoder_fn']
dec_final_fn = art['final']['decoder_fn']
ae_final_fn  = art['final']['autoencoder_fn']

# Call them on arbitrary NumPy arrays:
Z_new   = enc_final_fn(X_np[:1000])
Z_ns    = enc_mse_fn(X_np[:1000])
Xrec    = ae_final_fn(X_np[:1000])
Xrec_m  = dec_mse_fn(Z_new)  # decoder-only, etc.

m=15
color = X_np[:, 1]           # already numpy
plot_latent(
    Z_mse, color, s=0.01,           # both already numpy
    title=f"Latent representation before DCAE",
    out_path=os.path.join("/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS",
                          f"latent_mse_only_m{m}.png"))
plot_latent(
    Z_final, color, s=0.01,           # both already numpy
    title=f"Latent representation after DCAE",
    out_path=os.path.join("/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS",
                          f"latent_DCAE_m{m}.png"))