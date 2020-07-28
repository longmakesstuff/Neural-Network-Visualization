package de.longuyen.neuronalnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class Relu : Activation() {
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