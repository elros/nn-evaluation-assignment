# Task 1
- I've made some assumptions about the number of workers, for now using the CPU count. Ideally, this parameter should come from some kind of configuration to be more controllable.
- Otherwise, the task is solved with multiprocessing.Pool, with some optimizations to not let pandas read the whole file into the memory of the master process.
- Some time spent to find the torch features needed here. No LLM help for this task.


# Task 2
- Given the restricted time, I used ChatGPT to understand how torch and einops are typically used together. Then used this as inspiration to implement SimpleNeuralNetwork.backward().
- One minor thing: looks like the batch_size parameter wasn't used anymore, so had to be removed. Assuming I'm not missing something.
