## Code Usage

Our code consists of two folders: Neural_Network and Nonconvex_LR, where you will
find the full code for the respective problems. 

If you want to implement a short version, you may choose to run file run_demo.py 
in the Neural_Network folder. 

Our run_demo.py file implements our experiments for ONLY 40 EPOCHS. Then the results
are plotted automatically. You can modify the code if more epochs are needed. 

In the other hand, you can start with the run.py files in each folder, which contain
the hyper-parameter setting for each of our experiments. 

- Exp 1: Nonconvex Logistic Regression, diminishing LRs  (./Nonconvex_LR/run.py)
- Exp 2: Neural Network, diminishing LRs                 (./Neural_Network/run.py)
- Exp 3: Neural Network, three shuffling schemes         (./Neural_Network/run.py)

## Citation

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software please cite the following publication:

* Lam M. Nguyen, Quoc Tran-Dinh, Dzung T. Phan, Phuong Ha Nguyen, Marten van Dijk. **[A Unified Convergence Analysis for Shuffling-Type Gradient Methods](https://arxiv.org/abs/2002.08246)**. <em>arXiv preprint</em>, arXiv:2002.08246.

Feel free to send feedback and questions about the package to our maintainer Lam M. Nguyen at <LamNguyen.MLTD@gmail.com>.
