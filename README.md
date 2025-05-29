# dosrixs

**dosrixs** is a Python package for computing **resonant inelastic X-ray scattering** (RIXS) cross sections 
using projected dnesity of states (DOS) including polarization-dependent matrix elements which is based on the [Kramers-Heisenebrg formalism](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.59.2649).
It is designed to help researchers interpet
itinerant and charge-transfer features in experimental RIXS spectra. 
Our implementation is based on [M. Norman et al., Phys. Rev. B 107, 165124 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.165124).

## Features
- Compute RIXS spectra 
  
$$ \tilde{\sigma}(E_{\mathrm{in}}, E_{\mathrm{out}}, \epsilon) = \sum_{i,f,\sigma,\sigma',\epsilon^{\prime}} \int dE \rho_{f\sigma'}(E)\rho_{i\sigma}(E + E_{\mathrm{loss}}) \Big [ \frac{\Gamma}{2} \frac{M_{if\sigma\sigma'}(\epsilon,\epsilon')}{(E - E_{\mathrm{out}})^{2} + \Gamma^{2}/4}\Big ].$$

- Compute X-ray absoprtion spectra (XAS)

$$ \tilde{\sigma}(E_{\mathrm{in}},\epsilon) = \sum_{i,\sigma} \int dE \frac{\Gamma}{2} \frac{\rho_{i\sigma}M_{i\sigma}(\epsilon)}{(E_{\mathrm{in}}-E)^{2}+ \Gamma^{2}/4}.$$

For more details, please see [M. Norman et al., Phys. Rev. B 107, 165124 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.165124) for more details on the notation.

## Modular Design
  Our implementation is based on the abstract expansion of objects in spherical harmonic basis (``YlmExpansion``). This enables the computation of the polarization matrix elements, $M_{if\sigma\sigma'}(\epsilon,\epsilon')$, for arbitrary polarizations, initial/final states, and core states.

### Transition amplitudes and Polarizations

Within the dipole-dipole approximation, we need to compute the following matrix elements:

$$ \propto \sum_{c} \Big | \langle f|\epsilon_{\mathrm{out}} |c\rangle\langle c | \epsilon_{\mathrm{in}} | i\rangle \Big |^{2}$$

where:

* $\epsilon_{\mathrm{in}}$, $\epsilon_{\mathrm{out}}$ are the incoming and outgoing polarization vectors,
* $|c\rangle$ is a core state,
* $|i\rangle$, $|f\rangle$ are valence states.

### ``YlmExpansion``
The valence and core states, as well as the polarizations are all written as expansions in a spherical harmonic basis as:

$$ |\psi\rangle = \sum_{m,\sigma} c_{m,\sigma}Y_{\ell}^{m} $$

where:
* $\ell$ is the angular quantum number,
* $m \in [-\ell, \ell]$ is the magnetic quantum number,
* $\sigma \in \{\uparrow,\downarrow \}$ spins,
* $c_{m,\sigma} \in \mathbb{C}$ are expansion coefficients.

The ``YlmExpansion`` object is the foundation of the entire implementation. Internally, we use Python dictionaries as the data storage.

### Dipole transitions are Gaunt coefficients
The angular part of the dipole matrix element is calculated using Gaunt coefficients $G(\ell_{1}=1, m_{c}, \ell_{2}=1, m_{q}, \ell_{3}=2, m_{d})$:

$$ \langle Y_{\ell_{c}}^{m_{c}}| \hat{r_{q}} | Y_{\ell_{d}}^{m_{d}} \rangle = \int d\Omega Y_{\ell_{c}}^{m_{c}}(\theta,\phi) Y_{1}^{m_{q}}(\theta,\phi) Y_{\ell_{d}}^{m_{d}}(\theta,\phi) =G(m_{c},m_{q},m_{d})$$

## Contributing
We welcome contributions! Please open issues for bugs or feature requests, or submit pull requests.


## License

[MIT License](LICENSE)
