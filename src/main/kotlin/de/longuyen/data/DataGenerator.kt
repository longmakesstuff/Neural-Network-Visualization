package de.longuyen.data

interface DataGenerator {
    fun generate(): Pair<Array<IntArray>, Array<IntArray>>
}