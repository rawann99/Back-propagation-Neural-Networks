# Back-propagation-Neural-Networks
you are asked to write two programs. The first one implements the back
propagation algorithm and the second one implements the feed forward. In both assignments,
you will need to read the input (x vector) and the output (y vector) from a file. Below is the
structure of the input file:
1) First line: M, L, N where M is number of Input Nodes, L is number of Hidden Nodes and
N is number of Output Nodes
2) Second line: K, the number of training examples, each line has length M+N values, first
M values are X vector and last N values are output values. 3) K lines follow as described
An example of input file (just for clarification not to be used):
3 2 2
3
1 1 1.5 2 2
-1 2.25 0.5 -0.5 1.2
1 1 1 1 2
Above is a file that describes:
1) Network with 3 input nodes, 2 hidden and 2 output
2) Training is 3 examples
3) Second example has training example X [1 1 1.5] and output vector [2 2]
Deliverables of the first program:
• After reading the input file, you will perform back propagation algorithm, use dr. Notes
as reference. (The algorithm should stop after reaching an acceptable MSE, or after
running for a number of iterations, let’s say 500 iterations.)
• After finishing the 500 iterations, or reaching an acceptable MSE print that MSE on the
screen and save your final weights to a file.
Deliverables of the second program:
• After reading the input file, you will perform feed forward algorithm on the input data
using the best weights you have calculated in the first program.
• Print the MSE.
Note: we will give you the input file in the discussion, but as mentioned before it will have
the same structure as the input file of the first program.
