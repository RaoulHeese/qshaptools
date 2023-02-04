******************************
Quantum Shapley Value Toolbox
******************************

.. image:: https://img.shields.io/badge/license-MIT-lightgrey
    :target: https://github.com/RaoulHeese/qtree/blob/main/LICENSE
    :alt: MIT License

Experimental Python toolbox for Shapley values with uncertain value functions in general (see `Shapley Values with Uncertain Value Functions: arxiv.2301.08086 <https://doi.org/10.48550/arxiv.2301.08086>`_) and quantum Shapley values in particular (see `Explainable Quantum Machine Learning: arxiv.2301.09138 <https://doi.org/10.48550/arxiv.2301.09138>`_). For quantum Shapley values, the toolbox presumes a representation of quantum circuits via Qiskit.

**Usage**

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


📖 **Citation**

If you find this code useful in your research, please consider citing:

.. code-block:: tex

	@misc{https://doi.org/10.48550/arxiv.2301.09138,
		  doi = {10.48550/ARXIV.2301.09138}, 
		  url = {https://arxiv.org/abs/2301.09138},
		  author = {Heese, Raoul and Gerlach, Thore and Mücke, Sascha and Müller, Sabine and Jakobs, Matthias and Piatkowski, Nico},  
		  keywords = {Quantum Physics (quant-ph), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Physical sciences, FOS: Physical sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
		  title = {Explainable Quantum Machine Learning},
		  publisher = {arXiv},
		  year = {2023},
		  copyright = {arXiv.org perpetual, non-exclusive license}
		 }
