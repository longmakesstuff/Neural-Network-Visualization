package de.longuyen.neuralnetwork

import de.longuyen.neuralnetwork.activations.Activation
import de.longuyen.neuralnetwork.initializers.Initializer
import de.longuyen.neuralnetwork.losses.LossFunction
import de.longuyen.neuralnetwork.metrics.Metric
import de.longuyen.neuralnetwork.optimizers.Optimizer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.Serializable


/**
 * Vectorized deep neuronal network for supervised machine learning.
 *
 * @param layers Notate how many input each layer of the network will become. The first element of the array is the number
 * of the dataset's features. The last element of the array can be interpreted as the output of the network.
 *
 * @param hiddenActivation Each output of the hidden layers will be scaled with this function
 *
 * @param lastActivation Last output of the last hidden layer will be scaled with this function
 *
 * @param lossFunction Determined how the result of the network should be evaluated against a target.
 */
class NeuralNetwork(private val layers: IntArray, private val initializer: Initializer, private val hiddenActivation: Activation, private val lastActivation: Activation, private val lossFunction: LossFunction, private val optimizer: Optimizer, private val metric: Metric) : Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }
    /*
     * Intern parameters of the network
     */
    private val weights = initializer.initialize(layers)

    /*
     * Indicate how many hidden layers there are
     */
    private val hiddenCount = layers.size - 1


    /**
     * Adjusting model's parameters with the given data set.
     * @param X training features
     * @param Y training target
     * @param x validating features
     * @param y validating target
     * @param epochs how many epochs should the training take
     * @param verbose should the model print the current metric?
     * @param batchSize for faster training
     * @return history of the training with "val-loss", "train-loss", "val-metric", "train-metric"
     */
    fun train(X: INDArray, Y: INDArray, x: INDArray, y: INDArray, epochs: Long, batchSize: Long, verbose: Boolean=true): Map<String, DoubleArray>{
        val validationLosses = mutableListOf<Double>()
        val trainingLosses = mutableListOf<Double>()
        val trainingMetrics = mutableListOf<Double>()
        val validationMetrics = mutableListOf<Double>()

        for (epoch in 0 until epochs) {
            // Mini batch training
            for (i in 0 until X.shape()[0] step batchSize) {
                val Xi = X.get(NDArrayIndex.interval(0, x.shape()[0]), NDArrayIndex.interval(i, i + batchSize))
                val Yi = Y.get(NDArrayIndex.interval(0, y.shape()[0]), NDArrayIndex.interval(i, i + batchSize))

                train(Xi, Yi, Xi, Yi, 1, false)
            }

            // Summarize the epoch
            if (verbose) {
                // Calculate losses on training and validating dataset
                val validationPrediction = inference(x)
                val trainingPrediction = inference(X)
                val validationLoss = lossFunction.forward(yTrue = y, yPrediction = validationPrediction)
                val validationMetric = metric.compute(yTrue = y, yPrediction = validationPrediction)
                val trainingLoss = lossFunction.forward(yTrue = Y, yPrediction = trainingPrediction)
                val trainingMetric = metric.compute(yTrue = Y, yPrediction = trainingPrediction)
                println("Epoch $epoch - Training loss ${(trainingLoss.element() as Double)} - Validation Loss ${(validationLoss.element() as Double)} - Training Metric $trainingMetric - Validation Metric $validationMetric")

                validationLosses.add(validationLoss.element() as Double)
                trainingLosses.add(trainingLoss.element() as Double)
                validationMetrics.add(validationMetric)
                trainingMetrics.add(trainingMetric)
            }
        }
        val ret = mutableMapOf<String, DoubleArray>()
        ret["val-loss"] = validationLosses.toDoubleArray()
        ret["train-loss"] = trainingLosses.toDoubleArray()
        ret["val-metric"] = validationMetrics.toDoubleArray()
        ret["train-metric"] = trainingMetrics.toDoubleArray()
        return ret
    }

    /**
     * Adjusting model's parameters with the given data set.
     * @param X training features
     * @param Y training target
     * @param x validating features
     * @param y validating target
     * @param epochs how many epochs should the training take
     * @param verbose should the model print the current metric?
     * @return history of the training with "val-loss", "train-loss", "val-metric", "train-metric"
     */
    fun train(X: INDArray, Y: INDArray, x: INDArray, y: INDArray, epochs: Long, verbose: Boolean=true): Map<String, DoubleArray> {
        val validationLosses = mutableListOf<Double>()
        val trainingLosses = mutableListOf<Double>()
        val trainingMetrics = mutableListOf<Double>()
        val validationMetrics = mutableListOf<Double>()

        for (epoch in 0 until epochs) {
            // Forward propagation
            val cache = forward(X)

            // Backward propagation
            backward(X, Y, cache)

            if(verbose) {
                // Calculate losses on training and validating dataset
                val yPrediction = inference(x)
                val trainingLoss = lossFunction.forward(yTrue = Y, yPrediction = cache["A$hiddenCount"]!!)
                val validationLoss = lossFunction.forward(yTrue = y, yPrediction = yPrediction)
                val validationMetric = metric.compute(yTrue = y, yPrediction = yPrediction)
                val trainingMetric = metric.compute(yTrue = Y, yPrediction =  cache["A$hiddenCount"]!!)
                println("Epoch $epoch - Training loss ${(trainingLoss.element() as Double)} - Validation Loss ${(validationLoss.element() as Double)} - Training Metric $trainingMetric - Validation Metric $validationMetric")

                validationLosses.add(validationLoss.element() as Double)
                trainingLosses.add(trainingLoss.element() as Double)
                validationMetrics.add(validationMetric)
                trainingMetrics.add(trainingMetric)
            }
        }

        val ret = mutableMapOf<String, DoubleArray>()
        ret["val-loss"] = validationLosses.toDoubleArray()
        ret["train-loss"] = trainingLosses.toDoubleArray()
        ret["val-metric"] = validationMetrics.toDoubleArray()
        ret["train-metric"] = trainingMetrics.toDoubleArray()
        return ret
    }

    /**
     * Forward propagation and return the output of the last layer
     * @param x features
     * @return last output of the network
     */
    fun inference(x: INDArray): INDArray {
        return forward(x)["A$hiddenCount"]!!
    }

    /**
     * Forward propagation
     * @param x features
     * return the output of each layer and its corresponding scaled version
     */
    private fun forward(x: INDArray): MutableMap<String, INDArray> {
        val cache = mutableMapOf<String, INDArray>()

        // Calculate the output of the first layer manually
        cache["Z1"] = (weights["W1"]!!.mmul(x)).add(weights["b1"]!!)
        cache["A1"] = hiddenActivation.forward(cache["Z1"]!!)

        // Calculate the output of the next hidden layers atuomatically
        for (i in 2 until layers.size - 1) {
            cache["Z$i"] = (weights["W$i"]!!.mmul(cache["A${i - 1}"])).add(weights["b$i"]!!)
            cache["A$i"] = hiddenActivation.forward(cache["Z$i"]!!)
        }

        // Calculate the output of the last layer manually so I scale the output with another activation function
        cache["Z${layers.size - 1}"] = (weights["W${layers.size - 1}"]!!.mmul(cache["A${layers.size - 2}"])).add(weights["b${layers.size - 1}"]!!)
        cache["A${layers.size - 1}"] = lastActivation.forward(cache["Z${layers.size - 1}"]!!)

        return cache
    }

    /**
     * Backward propagation
     * @param x training features
     * @param y training target
     * @param cache the cached output of each layer, unscaled and scaled
     */
    private fun backward(x: INDArray, y: INDArray, cache: MutableMap<String, INDArray>) {
        val grads = mutableMapOf<String, INDArray>()

        // Calculate gradients of the loss respected to prediction
        grads["dZ$hiddenCount"] = lossFunction.backward(yTrue = y, yPrediction = cache["A$hiddenCount"]!!)

        // Calculate the gradients of loss respected to the output and the activated output
        for (i in 1 until hiddenCount) {
            grads["dA${hiddenCount - i}"] = (weights["W${hiddenCount - i + 1}"]!!.transpose()).mmul(grads["dZ${hiddenCount - i + 1}"]!!)
            grads["dZ${hiddenCount - i}"] = grads["dA${hiddenCount - i}"]!!.mul(hiddenActivation.backward(cache["Z${hiddenCount - i}"]!!))
        }

        // Calculate the gradients of loss respected to weights and biases
        grads["dW1"] = grads["dZ1"]!!.mmul(x.transpose())
        grads["db1"] = grads["dZ1"]!!.sum(true, 1)
        for (i in 2 until layers.size) {
            grads["dW$i"] = grads["dZ$i"]!!.mmul(cache["A${i - 1}"]!!.transpose())
            grads["db$i"] = grads["dZ$i"]!!.sum(true, 1)
        }

        optimizer.optimize(weights, grads, layers.size)
    }

    override fun toString(): String {
        return "NeuronalNetwork(layers=${layers.contentToString()}, initializer=$initializer, hiddenActivation=$hiddenActivation, lastActivation=$lastActivation, lossFunction=$lossFunction, optimizer=$optimizer)"
    }
}
