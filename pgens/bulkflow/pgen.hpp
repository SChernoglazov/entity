#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
#include <stdlib.h>
#endif // MPI_ENABLED

namespace user
{
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct UniformDist : public arch::SpatialDistribution<S, M> {
    UniformDist(const M& metric)
      : arch::SpatialDistribution<S, M> { metric } {}

    // to properly scale the number density, the probability should be normalized to 1
    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE; 
    }    
  };
  
  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M>
  {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temperature1, temperature2;

    inline PGen(const SimulationParams &p, const Metadomain<S, M> &global_domain)
      : arch::ProblemGenerator<S, M>(p)
      , temperature1 {p.template get<real_t>("setup.temperature")}
      , temperature2 {p.template get<real_t>("setup.temperature2",temperature1)}{};
    
    inline void InitPrtls(Domain<S, M> &domain)
    {
      // standard uniform injector
      //arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, temperature, {1, 2});

      // uniform distribution defined by non-uniform injector
      auto       edist1  = arch::Maxwellian<S, M>(domain.mesh.metric,
						domain.random_pool,
						temperature1);
      auto       edist2  = arch::Maxwellian<S, M>(domain.mesh.metric,
						domain.random_pool,
						temperature2);

      const auto sdist = UniformDist<S, M>(domain.mesh.metric);

      arch::InjectNonUniform<S, M, decltype(edist1), decltype(edist2), decltype(sdist)>(
            params,
            domain,
            { 1, 2 },
            { edist1, edist2 },
            sdist,
            1.0);

      for (std::size_t s { 0 }; s < 2; ++s) {
	auto& species = domain.species[s];
        auto  ux1     = species.ux1;
        auto  ux2     = species.ux2;
        auto  ux3     = species.ux3;
        auto  tag     = species.tag;

	std::cout << "species " << s << " mass " << species.mass() << " charge " << species.charge() <<std::endl;
	Kokkos::parallel_for(
          "CurvatureCooling",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }
	    ux1(p) = 0.75;
	    ux2(p) = 0.0;
	    ux3(p) = 0.0;
	  });
      }
    }
  };
} // namespace user

#endif
