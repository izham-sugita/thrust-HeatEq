#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<fstream>
#include<vector>
#include<chrono>

#include<blocksize.h> //contains blockDimX, blockDimY

#define pi 4.0*atan(1.0)

using namespace std;
using namespace std::chrono;

void   swap
// ====================================================================
//
// purpos     :  update the variable fn --> f
//
// date       :  Jul 03, 2001
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float   **f,        /* dependent variable                        */
   float   **fn        /* updated variable                          */
)
// --------------------------------------------------------------------
{
     float  *tmp = *f;   *f = *fn;   *fn = tmp;
}


//Heat equation kernel
__global__  void  HeatEq2D
// ====================================================================
//
// program    :  Two-dimensional heat equation kernel
//
// date       :  2018/09/15
// programmer :  Muhammad Izham 
// place      :  Universiti Malaysia Perlis
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* dependent variable                        */
   int      nx,         /* grid number in the x-direction            */
   int      ny,         /* grid number in the x-direction            */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
   int    j,    jx,   jy;
   float  fcc,  fce,  fcw,  fcs,  fcn;

   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;
   /*Dirichilet BC, fixed boundary*/
if(jx > 0  && jx < nx-1){
if(jy > 0 && jy < ny-1){
  j = nx*jy + jx;
     fcc = f[j];
     fcw = f[j - 1];
     fce = f[j+1];
     fcs = f[j-nx];
     fcn = f[j+nx];
     
   fn[j] = c0*(fce + fcw)
         + c1*(fcn + fcs)
         + c2*fcc;
 }
}

}

int main()
{

  int imax = 128;
  int jmax = 128;

  cout<<"Enter imax, jmax \n";
  cin>>imax>>jmax;
  
  float dx = 1.0f /(float)(imax-1);
  float dy = 1.0f /(float)(jmax-1);
  float dt = 0.01f*dx*dx;

  //test std::vector
  vector<float> h_test;
  h_test.resize(imax*jmax);
    for(int i=0; i<imax; ++i){
    for(int j=0; j<jmax; ++j){
      int id = i*jmax + j;
      h_test[id] = sin((float)i*pi*dx )*sin((float)j*pi*dy );
    }
  }
  
  thrust::host_vector<float> h_Told(imax*jmax);
  thrust::host_vector<float> h_Tnew(imax*jmax); 

  /*Initiate host vector*/
  for(int i=0; i<imax; ++i){
    for(int j=0; j<jmax; ++j){
      int id = i*jmax + j;
      h_Told[id] = sin((float)i*pi*dx )*sin((float)j*pi*dy );
      h_Tnew[id] = 0.0f;
    }
  }

  ofstream finit;
  finit.open("initHeat2D.csv");
  finit << "x, y, z, Temp\n";
  finit << setprecision(8);
  finit << fixed;
  for(int i=0; i<imax; ++i){
    for(int j=0; j<jmax; ++j){
      int id = i*jmax + j;
      finit<<(float)i*dx<<","
	<<(float)j*dy<<","
	<<h_test[id]<<","
	<<h_test[id]<<endl;
    }
  }
  finit.close();

  FILE *fp0;
  fp0 = fopen("initHeatOld.csv","w");
  fprintf(fp0,"x, y, z, temp\n");
  for(int i=0; i<imax; ++i){
    for(int j=0; j<jmax; ++j){
      int id = i*jmax + j;
      float xg = (float)i*dx;
      float yg = (float)j*dy;
      fprintf(fp0,"%f, %f, %f, %f\n", xg, yg, h_test[id], h_test[id]);
	}
  }
  fclose(fp0);
  
  thrust::device_vector<float> d_Told = h_Told;
  thrust::device_vector<float> d_Tnew = h_Tnew;
  
  float kappa = 1.0f;
  float c0 = kappa*dt/(dx*dx), c1 = kappa*dt/(dy*dy),
               c2 = 1.0 - 2.0*(c0 + c1);

  float *d_ToldPointer = thrust::raw_pointer_cast(&d_Told[0]);
  float *d_TnewPointer = thrust::raw_pointer_cast(&d_Tnew[0]);

  dim3 grid(imax/blockDimX, jmax/blockDimY, 1);
  dim3 threads(blockDimX, blockDimY, 1);
  
  int itermax =20000;
  double flops = 0.0f;

  high_resolution_clock::time_point t1=high_resolution_clock::now();  
  
  for(int iter=0; iter<itermax; ++iter){
  /*Calling kernel*/
  HeatEq2D<<<grid,threads>>>(d_ToldPointer,
				       d_TnewPointer,
				       imax,jmax,c0,c1,c2);

  
  //Update device_vector
  //d_Told = d_Tnew;

  swap(&d_ToldPointer,&d_TnewPointer); //the best way to swap!
  
  flops = flops + 7.0*((float)imax * (float)jmax);
  
  }/*end time loop*/

  high_resolution_clock::time_point t2=high_resolution_clock::now();

  duration<double> elapsed_time = duration_cast< duration<double> >(t2-t1);

  double timing = elapsed_time.count();

  cout<<"Total operations: "<<flops<<endl;

  flops = flops/(timing*1.0e9);
  cout<<"Elapsed time for "<<itermax<<" steps is "<<timing<<" secs."<<endl;
  cout<<"Performance: "<<flops<<" GFLOPS"<<endl;
  
  /*copy data back to host*/
  h_Tnew = d_Told;
  
  /*output data .csv*/
  ofstream fp;
  fp.open("thrustHeat.csv");
  fp << "x, y, z, Temp\n";
  fp << setprecision(8);
  fp << fixed;
  for(int i=0; i<imax; ++i){
    for(int j=0; j<jmax; ++j){
      int id = i*jmax + j;
      fp<<(float)i*dx<<","
	<<(float)j*dy<<","
	<<h_Tnew[id]<<","
	<<h_Tnew[id]<<endl;
    }
  }
  fp.close();
  
}
