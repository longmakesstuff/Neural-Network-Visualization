package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

/**
 * Leaky Relu for hidden layers' activations
 * f(x) = max(eps * x, x)
 */
class LeakyRelu(private val alpha: Double = 0.01) : Activation(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    override fun forward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for (yi in input.indices) {
            for (xi in input[yi].indices) {
                if (input[yi][xi] < 0) {
                    input[yi][xi] *= alpha
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }

    override fun backward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for (yi in input.indices) {
            for (xi in input[yi].indices) {
                if (input[yi][xi] > 0) {
                    input[yi][xi] = 1.0
                } else {
                    input[yi][xi] = alpha
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }
}