package de.longuyen.neuralnetwork.optimizers

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Optimizer {
    abstract fun optimize(weights: MutableMap<String, INDArray>, gradients: MutableMap<String, INDArray>, layers: Int)

    override fun toString(): String {
        return this.javaClass.simpleName
    }
}