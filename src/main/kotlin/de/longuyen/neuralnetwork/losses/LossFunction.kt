package de.longuyen.neuralnetwork.losses

import org.nd4j.linalg.api.ndarray.INDArray

abstract class LossFunction {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    abstract fun forward(yTrue: INDArray, yPrediction: INDArray): INDArray

    abstract fun backward(yTrue: INDArray, yPrediction: INDArray): INDArray

    override fun toString(): String {
        return "Loss Function {${this.javaClass.simpleName}}"
    }
}