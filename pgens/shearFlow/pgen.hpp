#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include <stdlib.h>
#endif // MPI_ENABLED

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t                     temperature1, temperature2;
    const real_t                     Lx, Ly, Lz;
    const real_t                     kx, amp; 
 
    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature1 { p.template get<real_t>("setup.temperature") }
      , temperature2 { p.template get<real_t>("setup.temperature2", temperature1) }
      ,	amp { p.template get<real_t>("setup.amp") }
      , Lx { global_domain.mesh().extent(in::x1).second -
             global_domain.mesh().extent(in::x1).first }	
      , Ly { global_domain.mesh().extent(in::x2).second -
             global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second -
             global_domain.mesh().extent(in::x3).first }
      , kx {(float)constant::TWO_PI/Lx} {};

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto c_q0 = params.template get<real_t>("scales.q0");
      const auto c_n0 = params.template get<real_t>("scales.n0");
      const auto c_v0 = params.template get<real_t>("scales.V0");
      std::cout << "q0=" << c_q0 << " n0=" << c_n0 << " v0=" << c_v0 <<std::endl;      
      arch::InjectUniformMaxwellians<S, M>(params, domain, ONE, {temperature1,temperature2}, { 1, 2 });

      auto amp_ {amp};
      auto kx_ {kx};
      auto& mechF = domain.fields.mechForce;
      auto metric = domain.mesh.metric;
      if constexpr (D == Dim::_2D) {
	  Kokkos::parallel_for(
   	    "addShift",
	    domain.mesh.rangeAllCells(),
	    Lambda(index_t i1, index_t i2) {
	      const auto i1_ = COORD(i1);
	      const auto i2_ = COORD(i2);
	      
	      const coord_t<Dim::_2D> xc2d{i1_ + HALF, i2_};
	      coord_t<Dim::_2D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      mechF(i1, i2, 0) = 0.0;
	      mechF(i1, i2, 1) = amp_ * math::sin(kx_ * xPh[0]);;
	      mechF(i1, i2, 2) = 0.0;
	    });
	}
      if constexpr (D == Dim::_3D) {
	  Kokkos::parallel_for(
   	    "addShift",
	    domain.mesh.rangeAllCells(),
	    Lambda(index_t i1, index_t i2, index_t i3) {
	      const auto i1_ = COORD(i1);
	      const auto i2_ = COORD(i2);
	      const auto i3_ = COORD(i3);
	      
	      const coord_t<Dim::_3D> xc3d{i1_ + HALF, i2_, i3_};
	      coord_t<Dim::_3D> xPh { ZERO };
	      metric.template convert<Crd::Cd, Crd::Ph>(xc3d, xPh);
	      mechF(i1, i2, i3, 0) = 0.0;
	      mechF(i1, i2, i3, 1) = amp_ * math::sin(kx_ * xPh[0]);;
	      mechF(i1, i2, i3, 2) = 0.0;
	    });
	}      
    }

  };
} // namespace user

#endif
