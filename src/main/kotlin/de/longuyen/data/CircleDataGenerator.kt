package de.longuyen.data

import de.longuyen.SIZE
import java.awt.Point
import java.util.*

class CircleDataGenerator(private val size: Int) : DataGenerator {
    override fun generate(): Pair<Array<IntArray>, Array<IntArray>> {
        val xs = mutableListOf<IntArray>()
        val ys = mutableListOf<IntArray>()
        val random = Random()
        val center = Point(SIZE / 2, SIZE / 2)

        var oneCounts = 0
        var zeroCounts = 0
        val half = size / 2
        while(oneCounts < half|| zeroCounts < half) {
            val x = random.nextInt(SIZE)
            val y = random.nextInt(SIZE)
            val radius = center.distance(Point(x, y))
            if(oneCounts < half && radius < 200){
                xs.add(intArrayOf(x, y))
                ys.add(intArrayOf(1))
                oneCounts += 1
            }else if(zeroCounts < half && radius > 200){
                xs.add(intArrayOf(x, y))
                ys.add(intArrayOf(0))
                zeroCounts += 1
            }
        }

        return Pair(xs.toTypedArray(), ys.toTypedArray())
    }
}