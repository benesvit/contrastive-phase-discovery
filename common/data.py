import numpy as np
import matplotlib.pyplot as plt


class DataWithLabels():
    """
    Configurations with everything stored alongside them: the model parameters
    they were generated at (values), descriptor vectors (x), energies and cluster
    labels, where -1 marks HDBSCAN noise.

    x, energies and labels are optional and are None when not available.
    """
    OPTIONAL_KEYS = {'x': 'X', 'energies': 'Energies', 'labels': 'Labels'}

    def __init__(self, configurations, values, x=None, energies=None, labels=None):
        self.configurations = configurations
        self.values = values
        self.x = x
        self.energies = energies
        self.labels = labels
        self.augumented = np.array([False]*len(configurations))
        self._check_lengths()

    def _check_lengths(self):
        n = len(self.configurations)
        for attr in ['values', *self.OPTIONAL_KEYS]:
            arr = getattr(self, attr)
            if arr is not None and len(arr) != n:
                raise ValueError(f"'{attr}' has length {len(arr)}, "
                                 f"expected {n} (number of configurations)")

    def _require(self, attr):
        if getattr(self, attr) is None:
            raise ValueError(f"'{attr}' is not available in this dataset")

    @classmethod
    def from_npz(cls, data_path):
        loaded = np.load(data_path)
        optional = {attr: loaded[key] if key in loaded.files else None
                    for attr, key in cls.OPTIONAL_KEYS.items()}
        return cls(loaded['Configurations'], loaded['Values'], **optional)

    def to_npz(self, data_path):
        optional = {key: getattr(self, attr)
                    for attr, key in self.OPTIONAL_KEYS.items()
                    if getattr(self, attr) is not None}
        np.savez(
            data_path,
            Configurations=self.configurations,
            Values=self.values,
            **optional
        )

    def _subset(self, idx):
        """Index every stored array, keeping missing ones as None."""
        return {attr: None if getattr(self, attr) is None else getattr(self, attr)[idx]
                for attr in ['configurations', 'values', *self.OPTIONAL_KEYS]}

    def filter_label(self, label):
        self._require('labels')
        return DataWithLabels(**self._subset(self.labels == label))

    def __getitem__(self, idx):
        s = self._subset(idx)
        return dict(Configuration=s['configurations'],
                    Value=s['values'],
                    X=s['x'],
                    Energy=s['energies'],
                    Label=s['labels'])
    
    def plot(self, idx, scale_eps = 1, vert: bool = False, figname: str = None, fig_kw: dict = {} ):
        P = self.configurations[idx][:, :,  :2]
        A = self.configurations[idx][:, :,  2:]
        norm_P = np.linalg.norm(P, axis=-1)
        norm_A = np.linalg.norm(A, axis=-1)
        max_P = np.max(norm_P)
        max_A = np.max(norm_A)

        scalep = 1. / np.max(norm_P) if max_P > scale_eps else 1.0
        scalea = 1. / np.max(norm_A) if max_A > scale_eps else 1.0

                # Prepare grid
        M, N, _ = P.shape
        x, y = np.meshgrid(np.arange(N), np.arange(M))

        # Recalculate pseudocubic coordinates to cartesian ones
        U, V = P[..., 0], P[..., 1]
        Eta, Xi = A[..., 0], A[..., 1]

        # Compute angles for each field (radians, range -pi to pi)
        angleP = np.arctan2(U, V)
        angleA = np.arctan2(Eta, Xi)

        # Common normalization and colormap for both fields
        norm = plt.Normalize(-np.pi, np.pi)
        cmap = 'hsv'

        # Create subplots
        if vert:
            fig, ax = plt.subplots(2, 1, sharex=True, **fig_kw)
        else:
            fig, ax = plt.subplots(1, 2, sharey=True, **fig_kw)

        q1 = ax[0].quiver(
            x, y, scalep * U, scalep * V, angleP,
            scale=1, scale_units='xy', angles='xy',
            cmap=cmap, norm=norm
        )
        ax[0].set_title(r"Polarization Field P(u, v)")
        ax[0].set_xlim((-1, N))
        ax[0].set_ylim((-1, M))
        ax[0].set_aspect('equal', adjustable='box')

        q2 = ax[1].quiver(
            x + 0.5, y + 0.5, scalea * Eta, scalea * Xi, angleA,
            scale=1, scale_units='xy', angles='xy',
            cmap=cmap, norm=norm
        )
        ax[1].set_title(r"Hidden Field A($\eta$, $\xi$)")
        ax[1].set_xlim((-1, N))
        ax[1].set_ylim((-1, M))
        ax[1].set_aspect('equal', adjustable='box')

        if figname:
            plt.savefig(figname, format="png", bbox_inches="tight")

        plt.tight_layout()
        plt.show()


    def set_labels(self, new_labels):
        self.labels = new_labels


    def _update_with_new(self, new_confs,
                    new_values,
                    new_xs=None,
                    new_energies=None,
                    new_labels=None,
                    augumented=True):
        self.configurations = np.concatenate([self.configurations, new_confs])
        self.values = np.concatenate([self.values, new_values],axis=0)
        for attr, new in [('x', new_xs), ('energies', new_energies), ('labels', new_labels)]:
            old = getattr(self, attr)
            if (old is None) != (new is None):
                raise ValueError(f"'{attr}' is present in only one of the datasets")
            if old is not None:
                setattr(self, attr, np.concatenate([old, new], axis=0))
        if augumented:
            self.augument = np.concatenate([self.augumented,
                                            np.array([True]*len(new_confs))])

    def augument_randomly(self, ratio : float = 0.5, repetition : bool = False):
        """
        Append randomly rotated copies of a random subset of the configurations.
        """
        rng = np.random.default_rng()
        n_all = len(self.configurations)
        n = int(n_all * ratio)

        idxs = rng.choice(range(n_all), n, replace=repetition)
        angles = rng.choice([90, 180, 270], n, replace=True)
            
        # rotate randomly chosen configurations
        new_confs = np.array([rotate(conf, dg) for (conf,dg) in zip(self.configurations[idxs], angles)])
        subset = self._subset(idxs)
        self._update_with_new(new_confs=new_confs,
                              new_values=subset['values'],
                              new_xs=subset['x'],
                              new_energies=subset['energies'],
                              new_labels=subset['labels'])


def rotate(field, degrees=90):
    """
    Rotate a configuration by a multiple of 90 degrees.

    Both the lattice and the vector components are rotated, since the fields
    transform with the lattice.

    Parameters
    ----------
    field : ndarray, shape (M, N, 4)
        Configuration with [...,:2] = P and [...,2:] = A.
    degrees : {90, 180, 270}

    Returns
    -------
    ndarray, shape (M, N, 4)
    """
    rotations = {
        90:  np.array([[0, -1], [1, 0]]),
        180: np.array([[-1, 0], [0, -1]]),
        270: np.array([[0, 1], [-1, 0]])
    }
    k = degrees // 90
    rot = rotations[degrees]
    P = field[..., :2] @ rot
    A = field[..., 2:] @ rot
    vector_rotated = np.dstack([P, A])
    return np.rot90(vector_rotated, k, axes=(0,1)).copy()