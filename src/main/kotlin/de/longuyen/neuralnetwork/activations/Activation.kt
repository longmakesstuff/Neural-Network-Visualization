package de.longuyen.neuralnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Activation{
    abstract fun forward(x: INDArray) : INDArray

    abstract fun backward(x: INDArray): INDArray

    override fun toString(): String {
        return "Activation function {${this.javaClass.simpleName}}"
    }
}