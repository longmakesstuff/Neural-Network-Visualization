package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable

/**
 * Tanh activation for hidden layers
 */
class Tanh: Activation(), Serializable{
    companion object {
        private const val serialVersionUID: Long = 1
    }

    override fun forward(x: INDArray): INDArray {
        val ez = Transforms.exp(x)
        val enz = Transforms.exp(x.mul(-1))
        return (ez.sub(enz)).div(ez.add(enz))
    }

    override fun backward(x: INDArray): INDArray {
        val tanh = Transforms.tanh(x)
        return Nd4j.ones(*x.shape()).sub(Transforms.pow(tanh, 2.0))
    }
}