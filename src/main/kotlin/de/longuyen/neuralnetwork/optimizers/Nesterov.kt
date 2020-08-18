package de.longuyen.neuralnetwork.optimizers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

class Nesterov(private val learningRate: Double, private val alpha: Double=0.1, private val gamma: Double=0.9) : Optimizer(), Serializable {
    companion object {
        private const val serialVersionUID: Long = 1
    }
    private val velocity = mutableMapOf<String, INDArray>()
    private var initialized = false



    override fun optimize(weights: MutableMap<String, INDArray>, gradients: MutableMap<String, INDArray>, layers: Int) {
        if(!initialized){
            initialized = true
            for(i in 2 until layers){
                velocity["vW$i"] = Nd4j.zerosLike(weights["W$i"]!!)
                velocity["vb$i"] = Nd4j.zerosLike(weights["b$i"]!!)
             }
        }
        for(i in 2 until layers){
            velocity["vW$i"] = velocity["vW$i"]!!.mul(gamma).add(gradients["dW$i"]!!.mul(alpha))
            weights["W$i"] = weights["W$i"]!!.sub(velocity["vW$i"]!!.mul(learningRate))

            velocity["vb$i"] = velocity["vb$i"]!!.mul(gamma).add(gradients["db$i"]!!.mul(alpha))
            weights["b$i"] = weights["b$i"]!!.sub(velocity["vb$i"]!!.mul(learningRate))
        }
    }

}