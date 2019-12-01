#include "hello_world.h"

__global__
void hello_world_kernel() {
    printf("Hello, world!\n");
}

void hello_world() {
    hello_world_kernel <<< 2, 2 >>> ();
    cudaDeviceSynchronize();
}