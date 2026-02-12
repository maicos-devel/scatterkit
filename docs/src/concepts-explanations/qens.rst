.. _qens-explanations:

============================
Quasi-elastic neutron scattering
============================


Quasi-elastic neutron scattering (QENS) is a powerful technique used to study the dynamics
of materials at the atomic and molecular level. In contrast to other neutron scattering
techniques, it provides insights into the motion of atoms and molecules, particularly in 
liquids, polymers, and biological systems. QENS is based on the scattering of neutrons by
the nuclei of the sample via the strong nuclear interactions. The term "quasi elastic"
refers to the situation where the energy transfer between the neutrons and the sample is
small compared to the incident neutron energy, thus allowing for the study of slow
dynamics and diffusive processes without perturbing the system significantly.

.. math::
    \Delta E \ll E_i

where :math:`\Delta E` is the energy transfer and :math:`E_i` is the incident neutron energy.

QENS experiments typically involve measuring the scattering intensity as a function of both
the momentum transfer :math:`\boldsymbol{q}` and the energy transfer :math:`\Delta E`. The
resulting data can be analyzed to extract information about the diffusion coefficients,
relaxation times, and other dynamic properties of the system under investigation. The technique
is particularly useful for studying phenomena such as molecular diffusion, rotational dynamics,
and collective motions in complex fluids, polymers, and biological macromolecules.

-----------
Experiments
-----------

In QENS experiments, a beam of neutrons is directed at the sample, and the scattered neutrons
are detected at various angles and energy transfers. The scattering intensity is measured as a
function of the momentum transfer :math:`\boldsymbol{q}` and the energy transfer :math:`\Delta E`.
The experimental setup typically includes a neutron source, a monochromator to select neutrons of
a specific energy, a sample holder, and a detector array to measure the scattered neutrons. The
data collected from QENS experiments can be analyzed to determine the dynamic properties of the
sample, such as diffusion coefficients and relaxation times.

-----------
Theory
-----------

The incident neutron beam can be described by a plane wave function, which can be expressed as

\Phi(\vec{r}, t) = A \cdot e^{i(\vec{k} \cdot \vec{r} - \omega t)}

where :math:`A` is the amplitude of the wave, :math:`\vec{k}` is the wave vector, and :math:`\omega`
is the angular frequency. The scattering of neutrons by the sample can be described by the
scattering function :math:`S(\boldsymbol{q}, \Delta E)`, which provides information about the dynamics
of the system. The scattering function can be related to the intermediate scattering function :math :`I(\boldsymbol{q}, t)` through a Fourier transform: