About
========

**dosrixs** is a Python package for computing **resonant inelastic X-ray scattering** (RIXS) cross sections 
using projected dnesity of states (DOS) including polarization-dependent matrix elements which is based on the `Kramers-Heisenebrg formalism <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.59.2649)>`_.
It is designed to help researchers interpet
itinerant and charge-transfer features in experimental RIXS spectra. 
Our implementation is based on `M. Norman et al., Phys. Rev. B 107, 165124 (2023) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.165124>`_.

- Compute RIXS spectra 
  
.. math:: \tilde{\sigma}(E_{\mathrm{in}}, E_{\mathrm{out}}, \epsilon) = \sum_{i,f,\sigma,\sigma',\epsilon^{\prime}} \int dE \rho_{f\sigma'}(E)\rho_{i\sigma}(E + E_{\mathrm{loss}}) \Big [ \frac{\Gamma}{2} \frac{M_{if\sigma\sigma'}(\epsilon,\epsilon')}{(E - E_{\mathrm{out}})^{2} + \Gamma^{2}/4}\Big ].

- Compute X-ray absoprtion spectra (XAS)

.. math:: \tilde{\sigma}(E_{\mathrm{in}},\epsilon) = \sum_{i,\sigma} \int dE \frac{\Gamma}{2} \frac{\rho_{i\sigma}M_{i\sigma}(\epsilon)}{(E_{\mathrm{in}}-E)^{2}+ \Gamma^{2}/4}.

For more details, please see `M. Norman et al., Phys. Rev. B 107, 165124 (2023) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.165124>`_ for more details on the notation.

