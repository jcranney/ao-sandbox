#!/usr/bin/env python
import numpy as np
import argparse
import aotools
import tqdm


def main():
    parser = argparse.ArgumentParser(
        "produce and save an ao dataset"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument(
        "--distribution", "-d", type=str, choices=["uniform", "vonkarman"],
        default="vonkarman", help="choice of distribution to generate data"
    )
    parser.add_argument(
        "--modes", "-m", type=int, default=100,
        help="number of modes to truncate dataset to"
    )
    parser.add_argument(
        "--imwidth", "-w", type=int, default=16,
        help="width of image in dataset (in pixels, nyquist sampled)"
    )
    parser.add_argument(
        "--dr0", type=float, default=1.0,
        help="diameter of the telescope divided by r0(lambda)"
    )
    parser.add_argument(
        "--pupdiam", type=int,
        default=32,  # pixels
        help="diameter of pupil in pixels (does not affect dataset size)"
    )
    parser.add_argument(
        "--out", "-o", type=str, default="out.npz",
        help="name of output file",
    )
    parser.add_argument(
        "--noise", help="flag for adding noise", action="count",
    )
    parser.add_argument(
        "--sampling", type=float, default=2.0,
        help="sampling of imager in units of FWHM (2.0 -> Nyquist)"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, default=1000,
        help="internal batch size for data generation (e.g., RAM limited)",
    )
    parser.add_argument(
        "--nbatches", "-n", type=int, default=1000,
        help="internal number of batches for data generation",
    )
    args = parser.parse_args()
    if args.command == "run":
        gen = generate_data(
            nmodes=args.modes, pupdiam=args.pupdiam,
            distribution=args.distribution, nsamples=args.batchsize,
            dr0=args.dr0, imwidth=args.imwidth, sampling=args.sampling,
        )
        modes, imgs = next(gen)
        for _ in tqdm.tqdm(range(args.nbatches)):
            modes_new, imgs_new = next(gen)
            modes = np.concatenate([modes, modes_new], axis=0)
            imgs = np.concatenate([imgs, imgs_new], axis=0)
        if args.noise:
            imgs = apply_noise(imgs)
        np.savez(
            args.out, modes=modes, imgs=imgs, args=args,
        )


def generate_data(*, nmodes: int, pupdiam: int, distribution: str,
                  nsamples: int, dr0: float, imwidth: int, sampling: float,
                  verbosity: int = 0):
    def log(message):
        if verbosity:
            print(message)
    log("initialising zernike modes")
    zernikes = aotools.zernikeArray(nmodes, pupdiam, norm="rms")
    pupil = (zernikes[0, :, :] == 1.0)  # the 0th zernike makes a great pupil
    half_pix_shift = np.array(np.mgrid[:pupdiam, :pupdiam]).sum(axis=0)
    half_pix_shift = -half_pix_shift/pupdiam*2*np.pi/4

    def modes_to_images(modes):
        log("projecting modes to phase space")
        phases = np.einsum("ijk,li->ljk", zernikes, modes) * pupil[None, :, :]
        log(f"rms wfe: {phases[:, pupil].std()} rad")
        log("converting phase to complex amplitude")
        psi = pupil[None, :, :] * \
            np.exp(1j*(phases + half_pix_shift[None, :, :]))
        log("imaging focal plane")
        fft_width = int(sampling*pupdiam)
        im_outer = (fft_width - imwidth)//2
        imgs = np.abs(
            np.fft.fftshift(np.fft.fft2(psi, s=[fft_width]*2), axes=[-2, -1])
        )**2 / (2*pupdiam)**2 / pupil.sum()
        log(f"mean uncropped image sum: {imgs.sum(axis=1).sum(axis=1).mean()}")
        imgs = imgs[:, im_outer:-im_outer, im_outer:-im_outer]
        return imgs

    if distribution == "vonkarman":
        log("building von Karman covariance matrix")
        yy, xx = np.meshgrid(
            np.arange(pupdiam)/pupdiam,
            np.arange(pupdiam)/pupdiam,
            indexing="ij",
        )
        xx = xx.flatten()
        yy = yy.flatten()
        rr = (
            (xx[:, None] - xx[None, :])**2 +
            (yy[:, None] - yy[None, :])**2
        ) ** 0.5
        cov = aotools.phase_covariance(rr, 1.0/dr0, 100.0)  # let L0 = 10.0*D
        log("eigen-decomposing covariance matrix")
        l, u = np.linalg.eigh(cov)
        cov_factor = u[:, l > 0] @ np.diag(l[l > 0]**0.5)
        log("inverting modes_to_phase mapping")
        modes_to_phase = zernikes[:, pupil].T
        phase_to_modes = np.linalg.solve(
            modes_to_phase.T @ modes_to_phase, modes_to_phase.T
        )

    while True:
        if distribution == "uniform":
            log("generating uniformly distributed random modes")
            modes = (np.random.random([nsamples, nmodes])-0.5)
            modes /= modes.std()
            modes *= (dr0)**(5/6) / nmodes**0.5
        elif distribution == "vonkarman":
            log("creating random phase screens")
            phases = np.einsum(
                "ij,lj->li",
                cov_factor,
                np.random.randn(nsamples, cov_factor.shape[1])
            ).reshape([nsamples, pupdiam, pupdiam])
            phases[:, pupil] -= phases[:, pupil].mean(axis=1)[:, None]
            log("projecting phase to modes")
            modes = (phase_to_modes @ phases[:, pupil].T).T
            log("filtering phase to modal space")
        yield modes, modes_to_images(modes)


def apply_noise(img, flux=200.0, ron=0.5):
    """Apply noise, with realistic defaults"""
    img = img * flux
    img = np.random.poisson(img)
    img = img + np.random.randn(*img.shape)*ron
    return img


def load_data(filename):
    data = np.load(filename)
    return data["modes"], data["imgs"]


if __name__ == "__main__":
    main()
