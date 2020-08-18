package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable

/**
 * Sigmoid function for scaling binary output
 */
class Sigmoid : Activation(), Serializable{
    companion object {
        private const val serialVersionUID: Long = 1
    }

    override fun forward(x: INDArray): INDArray {
        return Nd4j.ones(*x.shape()).div(Transforms.exp(x, true).add(1))
    }

    override fun backward(x: INDArray): INDArray {
        val e = this.forward(x)
        return e.mul(Nd4j.ones(*x.shape()).sub(e))
    }
}