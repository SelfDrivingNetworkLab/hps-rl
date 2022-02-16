import argparse
import os

import numpy as np
from scipy import signal

np.set_printoptions(precision=0, suppress=True, linewidth=150)


class Model:
    """
    MxM combining model

    M:          sqrt of number of input beams of square array
    b(x,y):     beam (complex ndarray(M,M))
    d(x,y):     diffractional optical element(doe) (complex ndarray(M,M))
    s(x,y):     superposition diffraction pattern (complex ndarray(N,N))
    b_f(u,v):   fft2d(b) in spatial-frequency domain (complex ndarray(M,M))
    d_f(u,v):   fft2d(d) in spatial-frequency domain (complex ndarray(N,N))
    s_f(u,v):   fft2d(s) in spatial-frequency domain (complex ndarray(N,N))
    """

    def __init__(self, **kwargs):
        self.M = kwargs.get("M", 3)
        self.beam_shape = (self.M, self.M)
        self.test_3_in_9 = kwargs.get('test_3_in_9', False)
        if self.M == 9:
            fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "doe81.npy")
            self.doe = kwargs.get("doe81", np.load(fname))
            self.Dn = self.M
        elif self.M == 3:
            if self.test_3_in_9:  # only test center 3x3 beams using 9x9 doe
                fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), "doe81.npy")
                doe = kwargs.get("doe81", np.load(fname))[6:-6, 6:-6]  # 15-6*2 = 3
                # normalize amplitude response to 1/M
                doe *= np.sqrt(1/(np.abs(doe)**2).sum())
                self.doe = doe
                self.Dn = self.M
            else:
                doe_phs_deg = np.array([0, 90, 0, 0, 0, -90, 180, 0, 0]).reshape(3, 3)
                self.Dn = np.sqrt(8)
                doe_amp = np.ones_like(doe_phs_deg) / self.Dn
                doe_amp[1, 1] = 0
                self.doe = doe_amp * np.exp(1j * np.deg2rad(doe_phs_deg))
        else:
            print("Invalid beam shape.")
            raise
        self.max_pwr = self.Dn ** 4
        self.rms_noise = kwargs.get("rms_noise", 0.1)
        h, w = self.doe.shape
        self.s_shape = (self.M + h - 1, self.M + w - 1)

        # higher order DOE shape v.s. input beam shape
        off = (self.doe.shape[0] - self.beam_shape[0]) // 2  # M=9:(15-9)/2, M=3:0
        self.doe1 = self.doe[off : off + self.M, off : off + self.M]
        self.reset()

    @property
    def beam_amp(self):
        return self._beam_amp

    @beam_amp.setter
    def beam_amp(self, value):
        assert value.shape == self.beam_shape
        self._beam_amp = value

    @property
    def beam_phs(self):
        return self._beam_phs

    @beam_phs.setter
    def beam_phs(self, value):
        assert value.shape == self.beam_shape
        self._beam_phs = value

    @property
    def pattern(self):
        if self._pattern is None:
            self.propagate()
        noise = np.random.randn(*self.s_shape) * self.rms_noise
        return np.clip(self._pattern + noise, 0, self.max_pwr)

    @property
    def combined_power(self):
        h, w = self.pattern.shape
        return self.pattern[h // 2, w // 2]

    @property
    def efficiency(self):
        h, w = self.pattern.shape
        return self.pattern[h // 2, w // 2] / self.pattern.sum() * 100

    @property
    def norm_eta(self):
        return self.efficiency / self._eta_ref

    def __repr__(self):
        return np.array2string(self.pattern, precision=0, max_line_width=150)

    def propagate(self):
        self._pattern = self.sim(self.beam_amp, self.beam_phs)

    def sim(self, beam_amp, beam_phs):
        s = self.conv2d(beam_amp, beam_phs)
        return np.abs(s * s.conj())

    def conv2d(self, beam_amp, beam_phs):
        """ s = b * d """
        b = beam_amp * np.exp(1j * beam_phs)
        return signal.convolve2d(b, self.doe)

    def deconv2d(self, s):
        """ B = S / D """
        M2 = self.M // 2
        doe_pad = np.pad(self.doe, ((M2, M2), (M2, M2)), "constant")
        d_f = np.fft.fft2(doe_pad)

        s_f = np.fft.fft2(s)
        kernel = np.fft.ifft2(s_f / d_f)
        kernel = np.fft.fftshift(kernel)
        off = self.doe.shape[0] // 2
        return kernel[off:-off, off:-off]

    def verify(self, beam_amp):
        # random input phase, forward simulation
        beam_phs = np.random.rand(*self.beam_shape) * np.pi * 2
        s = self.conv2d(beam_amp, beam_phs)
        # find b from s using d
        b = self.deconv2d(s)
        # compare with known beam array
        b_want = beam_amp * np.exp(1j * beam_phs)
        return np.allclose(b, b_want)

    def wrap_phase(self, phases):
        return (phases + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.beam_amp = np.rot90(np.rot90(np.abs(self.doe1))) * self.Dn ** 2
        if self.M == 9:
            # normalize to total input power of self.M**2
            scale = np.sqrt(np.sum(self.beam_amp ** 2)) / self.M ** 2
            self.beam_amp /= scale
        self.beam_phs = -np.rot90(np.rot90(np.angle(self.doe1)))
        self._pattern = self.propagate()
        self.beam_phs_ideal = self.beam_phs.copy()
        self.ideal_s_power = self.pattern
        self._eta_ref = self.efficiency

    def perturb(self, phs_deg=10, np_random=None):
        if np_random is None:
            np_random = np.random.RandomState()
        phs_pertub_deg = np_random.uniform(low=-phs_deg, high=phs_deg, size=self.beam_shape)
        self.beam_phs += np.deg2rad(phs_pertub_deg)
        self.propagate()


class Controller:
    def __init__(self, **kwargs):
        self.model = Model(**kwargs)
        self._gain = kwargs.get("gain", 1)
        self.reset_drive()

    def reset_drive(self):
        self._u = np.zeros(self.model.beam_shape)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        assert value.shape == self.model.beam_shape
        self._u = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    def correct(self, phs_err):
        u_delta = self.gain * phs_err
        self.u -= u_delta
        self.model.beam_phs -= u_delta
        self.model.propagate()

    def iterate(self):
        self.correct(0)

    def diagnose(self):
        return None


def main(m, test_3_in_9=False):
    model = Model(M=m, test_3_in_9=test_3_in_9)
    print("Asserting de-convolution...")
    beam_amp = np.random.rand(m, m)
    assert model.verify(beam_amp) is True

    np.set_printoptions(precision=1, suppress=True, linewidth=150)
    print("DOE amp response * {}: (expect 1)".format(m))
    print(np.abs(model.doe) * m)
    print("Input power:")
    print(model.beam_amp ** 2)
    print("Total Input power: {:.1f}".format(np.sum(model.beam_amp ** 2)))
    print("Ouput power:")
    print(model)
    print("Combined center: {:.1f}".format(model.combined_power))
    print("Efficiency: {:.3f} %".format(model.efficiency))
    # print(model)

    rms_deg = 10
    print("Applying perturbation...RMS {} deg per step".format(rms_deg))
    for i in range(20):
        model.perturb()
        print("Normalized Efficiency: {:.1f} %".format(100 * model.norm_eta))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-m", type=int, default=3, choices=[3, 9], help="MxM beam shape")
    p.add_argument("--test_3_in_9", action="store_true", help="test center 3x3 using 9x9 doe")
    args = p.parse_args()
    main(args.m, args.test_3_in_9)
