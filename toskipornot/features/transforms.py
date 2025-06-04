from monai.transforms import Randomizable, MapTransform, Transform
from monai.config import KeysCollection
from typing import Optional, Any, Mapping, Hashable
import numpy as np
import monai


"""
To add 'salt & pepper' noise with density d to an image, imnoise first assigns each pixel a random probability value 
from a standard uniform distribution on the open interval (0, 1).
For pixels with probability value in the range (0, d/2), the pixel value is set to 0. The number of pixels that are 
set to 0 is approximately d*numel(I)/2.
For pixels with probability value in the range [d/2, d), the pixel value is set to the maximum value of the image 
data type. The number of pixels that are set to the maximum value is approximately d*numel(I)/2.
For pixels with probability value in the range [d, 1), the pixel value is unchanged.
"""
class RandSaltAndPepperNoise(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, density: float = 0.05) -> None:
        self.prob = np.clip(prob, 0.0, 1.0)
        self.density = density
        self._noise = 0

    def randomize(self, data: np.ndarray) -> None:
        self._noise = 0

        if self.R.random() < self.prob:
            noise_array = self.R.rand(*data.shape[1:])[None]
            self._noise = noise_array.astype(data.dtype)

    def add_noise(self, img: np.ndarray) -> np.ndarray:
        typeinfo = np.iinfo(img.dtype)
        img[self._noise < self.density / 2] = typeinfo.min
        img[self.density / 2 < self._noise < self.density] = typeinfo.max
        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self.randomize(img)
        return self.add_noise(img)


class RandSaltAndPepperNoised(Randomizable, MapTransform):
    def __init__(
        self, keys: KeysCollection, prob: float = 0.5, density: float = 0.05
    ) -> None:
        super(Randomizable, self).__init__(keys)
        self.transform = RandSaltAndPepperNoise(prob, density)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSaltAndPepperNoised":
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self.transform.randomize(data)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Mapping[Hashable, np.ndarray]:
        self.randomize(data[monai.utils.first(self.keys)])

        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.add_noise(d[key])
        return d