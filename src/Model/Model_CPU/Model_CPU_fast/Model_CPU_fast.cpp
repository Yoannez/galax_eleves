#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

// OMP  version
/*#pragma omp parallel for
    for (int i = 0; i < n_particles; i ++){
        for (int j = 0; j < n_particles; j++){particles
            if(i != j){
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
        }
     }
articles.x[j] - particles.x[i
#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy batch<T, A> const &y) n [i] * 0.1f;
	}

*/
// OMP + xsimd version
#pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {	
		// load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
        for (int j = 0; j < n_particles; j++)
        {
			if(j>i-1 && j<i+b_type::size){
				for(int x=i; x<i+b_type::size; x++){
					if(x != j){
						const float diffx = particles.x[j] - particles.x[x];
						const float diffy = particles.y[j] - particles.y[x];
						const float diffz = particles.z[j] - particles.z[x];

						float dij = diffx * diffx + diffy * diffy + diffz * diffz;

						if (dij < 1.0)
						{
							dij = 10.0;
						}
						else
						{
							dij = std::sqrt(dij);
							dij = 10.0 / (dij * dij * dij);
						}

						accelerationsx[x] += diffx * dij * initstate.masses[j];
						accelerationsy[x] += diffy * dij * initstate.masses[j];
						accelerationsz[x] += diffz * dij * initstate.masses[j];
					}
				}
				raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        		raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        		raccz_i = b_type::load_unaligned(&accelerationsz[i]);	
			}else{
				// load registers body j
				const b_type rposx_j = b_type::broadcast(particles.x[j]);
				const b_type rposy_j = b_type::broadcast(particles.y[j]);
				const b_type rposz_j = b_type::broadcast(particles.z[j]);
				const b_type rmas_j = b_type::broadcast(initstate.masses[j]);

				b_type diffx = rposx_j - rposx_i;
				b_type diffy = rposy_j - rposy_i;
				b_type diffz = rposz_j - rposz_i;

				b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

				b_type dij_rsqrt = xs::sqrt(dij);
				b_type dij_real = 10.0 / (dij_rsqrt * dij_rsqrt * dij_rsqrt);

				dij = xs::fmin( b_type::broadcast(10.0), dij_real);

				raccx_i += diffx * dij * rmas_j;
				raccy_i += diffy * dij * rmas_j;
				raccz_i += diffz * dij * rmas_j;

                raccx_i.store_unaligned(&accelerationsx[i]);
                raccy_i.store_unaligned(&accelerationsy[i]);
                raccz_i.store_unaligned(&accelerationsz[i]);
			}
        }    
    }
#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

}

#endif // GALAX_MODEL_CPU_FAST
