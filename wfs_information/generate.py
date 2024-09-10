#!/usr/bin/env python
import numpy as np
import argparse
import aotools


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
        "--r0", type=float, default=1.0,
        help="seeing [m] (Fried parameter) at wavelength of 0.5 microns"
    )
    parser.add_argument(
        "--wavelength", type=float,
        default=1.55,  # microns
        help="wavefront sensor imaging wavelength in microns"
    )
    parser.add_argument(
        "--pupdiam", type=int,
        default=32,  # pixels
        help="diameter of pupil in pixels (does not affect dataset size)"
    )
    parser.add_argument(
        "--nsamples", "-s", type=int, default=10_000,
        help="number of samples/training pairs to include in dataset"
    )
    parser.add_argument(
        "--out", "-o", type=str, default="out.npz",
        help="name of output file",
    )
    parser.add_argument(
        "--teldiam", type=float, default=8.0,
        help="physical diameter of telescope [m]"
    )
    parser.add_argument(
        "--outerscale", type=float, default=25.0,
        help="outer scale of turbulence (a.k.a., L0)"
    )
    parser.add_argument(
        "--noise", help="flag for adding noise", action="count",
    )
    args = parser.parse_args()
    if args.command == "run":
        modes, imgs = make_data(
            nmodes=args.modes, pupdiam=args.pupdiam,
            distribution=args.distribution, nsamples=args.nsamples,
            teldiam=args.teldiam, r0=args.r0, L0=args.outerscale,
            wavelength=args.wavelength, imwidth=args.imwidth,
        )
        if args.noise:
            imgs = apply_noise(imgs)
        np.savez(
            args.out, modes=modes, imgs=imgs, args=args,
        )


def make_data(*, nmodes: int, pupdiam: int, distribution: str, nsamples: int,
              teldiam: float, r0: float, L0: float, wavelength: float,
              imwidth: int):
    print("creating zernike modes")
    zernikes = aotools.zernikeArray(nmodes, pupdiam, norm="rms")
    pupil = (zernikes[0, :, :] == 1.0)  # the 0th zernike makes a great pupil
    if distribution == "uniform":
        print("generating uniformly distributed random modes")
        modes = (np.random.random([nsamples, nmodes])-0.5)
        modes /= modes.std()
        modes *= (teldiam / r0)**(5/6) / nmodes**0.5
        print("projecting modes to phase space")
        phases = np.einsum("ijk,li->ljk", zernikes, modes) * pupil[None, :, :]
    elif distribution == "vonkarman":
        print("building von Karman covariance matrix")
        yy, xx = np.meshgrid(
            np.arange(pupdiam)/pupdiam*teldiam,
            np.arange(pupdiam)/pupdiam*teldiam,
            indexing="ij",
        )
        xx = xx.flatten()
        yy = yy.flatten()
        rr = (
            (xx[:, None] - xx[None, :])**2 +
            (yy[:, None] - yy[None, :])**2
        ) ** 0.5
        cov = aotools.phase_covariance(rr, r0, L0)
        print("eigen-decomposing covariance matrix")
        l, u = np.linalg.eigh(cov)
        cov_factor = u[:, l > 0] @ np.diag(l[l > 0]**0.5)
        print("creating random phase screens")
        phases = np.einsum(
            "ij,lj->li",
            cov_factor, np.random.randn(nsamples, cov_factor.shape[1])
        ).reshape([nsamples, pupdiam, pupdiam])
        print("inverting modes_to_phase mapping")
        modes_to_phase = zernikes[:, pupil].T
        phase_to_modes = np.linalg.solve(
            modes_to_phase.T @ modes_to_phase, modes_to_phase.T
        )
        print("projecting phase to modes")
        modes = (phase_to_modes @ phases[:, pupil].T).T
        print("filtering phase to modal space")
        # phases[:, pupil] = (modes_to_phase @ modes.T).T
        # phases *= pupil
    else:
        raise ValueError(f"Distribution mode not supported: {distribution}")
    print(f"rms wfe: {phases[:, pupil].std()} rad")
    print("converting phase to complex amplitude") 
    half_pix_shift = np.array(np.mgrid[:pupdiam, :pupdiam]).sum(axis=0)
    half_pix_shift = -half_pix_shift/pupdiam*2*np.pi/4
    psi = pupil[None, :, :]*np.exp(1j*(0.5/wavelength*phases + half_pix_shift))
    print("imaging focal plane")
    im_outer = (pupdiam*2 - imwidth)//2
    imgs = np.abs(
        np.fft.fftshift(np.fft.fft2(psi, s=[2*pupdiam]*2), axes=[-2, -1])
    )**2 / (2*pupdiam)**2 / pupil.sum()
    print(f"mean uncropped image sum: {imgs.sum(axis=1).sum(axis=1).mean()}")
    imgs = imgs[:, im_outer:-im_outer, im_outer:-im_outer]
    return modes, imgs


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
