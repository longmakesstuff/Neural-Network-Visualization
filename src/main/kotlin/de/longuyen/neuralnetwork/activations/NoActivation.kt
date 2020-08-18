package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable
/**
 * None activation for output layer. Used for regression purpose.
 */
class NoActivation : Activation(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    override fun forward(x: INDArray): INDArray {
        return x.dup()
    }

    override fun backward(x: INDArray): INDArray {
        return Nd4j.ones(*x.shape())
    }
}