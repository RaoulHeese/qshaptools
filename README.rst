******************************
Quantum Shapley Value Toolbox
******************************

.. image:: https://img.shields.io/badge/license-MIT-lightgrey
    :target: https://github.com/RaoulHeese/qtree/blob/main/LICENSE
    :alt: MIT License
	
.. image:: https://github.com/RaoulHeese/qshaptools/blob/master/_static/qshap.png?raw=true
    :alt: Title

Experimental Python toolbox for Shapley values with uncertain value functions in general (see `Shapley Values with Uncertain Value Functions <https://doi.org/10.48550/arxiv.2301.08086>`_) and quantum Shapley values in particular (see `Explaining Quantum Circuits with Shapley Values: Towards Explainable Quantum Machine Learning <https://doi.org/10.48550/arxiv.2301.09138>`_). Quantum Shapley values provide a method to measure the influence of gates within quantum circuit with respect to a freely customizable value function, for example expressibility or entanglement capability.


**Usage**

For quantum Shapley values, the toolbox presumes a representation of quantum circuits via Qiskit.

Minimal working example:

.. code-block:: python

    from qiskit import Aer
    from qiskit.utils import QuantumInstance
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.opflow import PauliSumOp
    from qshap import QuantumShapleyValues
    from qvalues import value_H
    from tools import visualize_shapleys

    # define circuit
    H = PauliSumOp.from_list([('ZZI', 1), ('ZII', 2), ('ZIZ', -3)])
    qc = QAOAAnsatz(cost_operator=H, reps=1)
    qc = qc.decompose().decompose().decompose()
    qc = qc.assign_parameters([0]*len(qc.parameters))

    # define quantum instance
    quantum_instance = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))

    # setup quantum Shapley values
    qsv = QuantumShapleyValues(qc, value_fun=value_H, value_kwargs_dict=dict(H=H), quantum_instance=quantum_instance)
    print(qsv)

    # evaluate quantum Shapley values
    qsv()

    # show results
    print(qsv.phi_dict)
    visualize_shapleys(qc, phi_dict=qsv.phi_dict).draw()

As a result, the quantum Shapley values assigned to each gate are plotted:

.. image:: https://github.com/RaoulHeese/qshaptools/blob/master/_static/output.png?raw=true
    :alt: Output


📖 **Citation**

If you find this code useful in your research, please consider citing `Explaining Quantum Circuits with Shapley Values: Towards Explainable Quantum Machine Learning <https://doi.org/10.48550/arxiv.2301.09138>`_:

.. code-block:: tex

    @misc{heese2023explaining,
      title={Explaining Quantum Circuits with Shapley Values: Towards Explainable Quantum Machine Learning}, 
      author={Raoul Heese and Thore Gerlach and Sascha Mücke and Sabine Müller and Matthias Jakobs and Nico Piatkowski},
      year={2023},
      eprint={2301.09138},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
      }

*This project is currently not under development and is not actively maintained.*
