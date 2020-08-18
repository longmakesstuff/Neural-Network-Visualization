package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

/**
 * Relu function for hidden layers' activation
 * f(x) = max(0, x)
 */
class Relu : Activation(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1032640494287127350
    }

    override fun forward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] < 0){
                    input[yi][xi] = 0.0
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }

    override fun backward(x: INDArray): INDArray {
        val input: Array<DoubleArray> = x.toDoubleMatrix()
        for(yi in input.indices){
            for(xi in input[yi].indices){
                if(input[yi][xi] > 0){
                    input[yi][xi] = 1.0
                }else{
                    input[yi][xi] = 0.0
                }
            }
        }
        return Nd4j.createFromArray(input).castTo(x.dataType())
    }
}