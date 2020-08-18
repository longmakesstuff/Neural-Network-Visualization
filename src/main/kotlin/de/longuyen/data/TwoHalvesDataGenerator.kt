package de.longuyen.data

import de.longuyen.SIZE
import java.util.*


class TwoHalvesDataGenerator(private val size: Int) : DataGenerator{
    override fun generate(): Pair<Array<IntArray>, Array<IntArray>> {
        val xs = mutableListOf<IntArray>()
        val ys = mutableListOf<IntArray>()
        val random = Random()
        val x1 = 0
        val y1 = SIZE
        val x2 = SIZE
        val y2 = 0
        for(i in 0 until size){
            val x = random.nextInt(SIZE)
            val y = random.nextInt(SIZE)
            xs.add(intArrayOf(x, y))
            val d = (x - x1)*(y2 - y1) - (y - y1) * (x2 - x1)
            if(d <= 0){
                ys.add(intArrayOf(1))
            }else{
                ys.add(intArrayOf(0))
            }
        }

        return Pair(xs.toTypedArray(), ys.toTypedArray())
    }

}