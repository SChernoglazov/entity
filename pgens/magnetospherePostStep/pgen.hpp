#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

  private:
    const real_t Bsurf, Rstar;
  };

  template <Dimension D>
  struct BoundaryFields {
    BoundaryFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Bsurf, Rstar;
  };
  
  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar, real_t omega,  real_t comp, real_t spinup)
      : InitFields<D> { bsurf, rstar }
      , time { time }
      , Omega { omega }
      , comp {comp}
      , spinup {spinup}
      , OmegaLT { static_cast<real_t>(0.4) * Omega * comp }{}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      real_t factor;
      if (time>spinup){
	factor = 1.0;
      }else{
	factor = time/spinup;
      }
      return (Omega - OmegaLT) * factor * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      //real_t factor = 1.0;
      real_t factor; 
      if (time>spinup){
	factor = 1.0;
      }else{
	factor = time/spinup;
      }
      return -(Omega - OmegaLT) * factor * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega, OmegaLT, comp, spinup;
  };

  Inline auto randf(index_t p) -> real_t {
    const index_t n = (1664525 * p + 1013904223);
    return (n & 0xFFFFFF) / static_cast<real_t>(0x1000000);
  }
  
  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t  Bsurf, Rstar, Omega;
    const real_t  Curv; 
    const real_t  RLC;
    const real_t  inv_n0, dt;    
    InitFields<D> init_flds;

    // these two lines are related to number density computation
    bool          is_first_step;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , RLC {1/Omega}
      , inv_n0 {ONE / p.template get<real_t>("scales.n0")}
      , dt { params.template get<real_t>("algorithms.timestep.dt") }
      , Curv {p.template get<real_t>("setup.Curvature")}
      , is_first_step { true }
      , init_flds { Bsurf, Rstar } {}
    
    inline PGen() {}

    /*void CustomFieldEvolution(std::size_t step, long double time, Domain<S, M>& domain, bool updateE, bool updateB) {
      if(updateB){
        const auto comp {params.template get<real_t>("setup.Compactness") };
        const auto _omega {static_cast<real_t>(constant::TWO_PI) / params.template get<real_t>("setup.period", ONE)};	
        const auto _rstar {Rstar};
        const auto _bsurf {Bsurf};
        const auto coeff { HALF * params.template get<real_t>
            ("algorithms.timestep.correction") * dt };
        auto& EB = domain.fields.em;
        auto metric = domain.mesh.metric;

        auto factor = 1.0;
        Kokkos::parallel_for(
          "addShift",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2) {
            real_t corrEx2iP1j, corrEx2ij, corrEx1ijP1, corrEx1ij;
            const auto i1_ = COORD(i1);
            const auto i2_ = COORD(i2);
            { // Etheta(i+1, j)
              const coord_t<Dim::_2D> xc2d{i1_+ONE, i2_ + HALF};
              coord_t<Dim::_2D> xPh { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
              auto shift = factor * static_cast<real_t>(0.4) * _omega * comp *
                CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);

              auto Bphys = _bsurf * math::cos(xPh[1]) / CUBE(xPh[0] / _rstar);
              auto Etheta = Bphys*shift;
              corrEx2iP1j = metric.template transform<2, Idx::T, Idx::U>({ i1_+ONE, i2_ + HALF },Etheta);
            }
            { // Etheta(i, j)
              const coord_t<Dim::_2D> xc2d{i1_, i2_ + HALF};   
              coord_t<Dim::_2D> xPh { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
              auto shift = factor * static_cast<real_t>(0.4) * _omega * comp *
                CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);
	      
              auto Bphys = _bsurf * math::cos(xPh[1]) / CUBE(xPh[0] / _rstar);
              auto Etheta = Bphys * shift;
              corrEx2ij = metric.template transform<2, Idx::T, Idx::U>({ i1_, i2_ + HALF },Etheta);
            }
            { // Er(i, j)
              const coord_t<Dim::_2D> xc2d{i1_ + HALF, i2_};               
              coord_t<Dim::_2D> xPh { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
              auto shift = factor * static_cast<real_t>(0.4) * _omega * comp *
                CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);
              auto Bphys = _bsurf * HALF * math::sin(xPh[1]) / CUBE(xPh[0] / _rstar); 
              auto Er = Bphys * shift; 
              corrEx1ij = metric.template transform<1, Idx::T, Idx::U>({ i1_ + HALF, i2_ },Er); 
            }
            { // Er(i, j+1)
              const coord_t<Dim::_2D> xc2d{i1_+HALF, i2_+ONE};   
              coord_t<Dim::_2D> xPh { ZERO };
              metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
              auto shift = factor * static_cast<real_t>(0.4) * _omega * comp *
                CUBE(_rstar / xPh[0]) * xPh[0] * math::sin(xPh[1]);
              auto Bphys = _bsurf * HALF * math::sin(xPh[1]) / CUBE(xPh[0] / _rstar); 
              auto Er = Bphys * shift; 
              corrEx1ijP1 = metric.template transform<1, Idx::T, Idx::U>({ i1_ + HALF, i2_ + ONE},Er); 
            }
            const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
                                         { i1_ + HALF, i2_ + HALF }) };
            const real_t h1_pHp1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + ONE }) };
            const real_t h1_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
            const real_t h2_p1pH { metric.template h_<2, 2>({ i1_ + ONE, i2_ + HALF }) };
            const real_t h2_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };

            //minus included into coeff 
            EB(i1, i2, em::bx3) += coeff* inv_sqrt_detH_pHpH *( h2_p1pH * corrEx2iP1j - h2_0pH * corrEx2ij + h1_pHp1 * corrEx1ijP1 - h1_pH0 * corrEx1ij);
          });
      }
      }*/

    
    auto AtmFields(real_t time) const -> DriveFields<D> {
      const auto comp {params.template get<real_t>("setup.Compactness") };
      const auto spinup {params.template get<real_t>("setup.spinup_time") };
      return DriveFields<D> { time, Bsurf, Rstar, Omega, comp, spinup };
    }

    //auto MatchFields(real_t) const -> InitFields<D> {
    auto MatchFields(real_t) const -> BoundaryFields<D> {
      return BoundaryFields<D> { Bsurf, Rstar };  
      //return InitFields<D> { Bsurf, Rstar };
    }

    void CustomPostStep(timestep_t, simtime_t time, Domain<S, M>& domain) {
      const auto gamma_thres1 {params.template get<real_t>("setup.GammaThres1")};
      const auto gamma_thres2 {params.template get<real_t>("setup.GammaThres2")};
      const auto stepsEmit {dt/params.template get<real_t>("setup.TimeEmit")};
      const auto angThres {params.template get<real_t>("setup.angThres") };
      const auto rad_radius {params.template get<real_t>("setup.rad_radius") };
      const auto PairDensity1 {params.template get<real_t>("setup.pair_dens1") };
      const auto PairDensity2 {params.template get<real_t>("setup.pair_dens2") };
      const auto PairDensityCS {params.template get<real_t>("setup.pair_densCS") };
      const auto gammaSec {params.template get<real_t>("setup.gammaSec") };
      const auto FlippingFraction {params.template get<real_t>("setup.flipFrac")};
      const auto FlippingDistance {params.template get<real_t>("setup.flipDist")};
      const auto PPinCS {params.template get<bool>("setup.PPinCS")};
      //const auto dt_ {dt};
      //const auto _Bsurf { Bsurf };
      //const auto Rstar_ { Rstar };
      const auto PolarCap { 0.7*math::sqrt(1.0/RLC) };

      real_t gamma_thres =  gamma_thres1;

      real_t PairDensity;
      if(time<20.0){
	PairDensity = PairDensity1;
      }else{
	PairDensity = PairDensity2;
      }
      
      auto metric = domain.mesh.metric;
      auto random_pool    = domain.random_pool;
      auto EB             = domain.fields.em; 

      //------------curvature cooling of pairs and curvature photons emission--------------
      	

      Kokkos::deep_copy(domain.fields.bckp, ZERO);
      auto scatter_bckp = Kokkos::Experimental::create_scatter_view(
							      domain.fields.bckp);
      const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");
      const auto ni2         = domain.mesh.n_active(in::x2);
      const auto use_weights = M::CoordType != Coord::Cart;
        
      for( int i=0; i<4; i++){
	auto& part = domain.species[i];
	if (part.mass() == 0){
	  continue;
	}
	if (part.npart()>0){
	  Kokkos::parallel_for(
            "ComputeMoments",
	    part.rangeActiveParticles(),
	    kernel::ParticleMoments_kernel<SimEngine::SRPIC, M, FldsID::N, 6>(
              {}, scatter_bckp, 0,
	      part.i1, part.i2, part.i3,
	      part.dx1, part.dx2, part.dx3,
	      part.ux1, part.ux2, part.ux3,
	      part.phi, part.weight, part.tag,
	      part.mass(), part.charge(),
	      use_weights,
	      domain.mesh.metric, domain.mesh.flds_bc(),
	      ni2, inv_n0, ONE));
	  part.set_unsorted();
	}
      }
      Kokkos::Experimental::contribute(domain.fields.bckp, scatter_bckp);
      auto BCKP = domain.fields.bckp;      

      for (std::size_t s { 0 }; s < 4; ++s) {

	if(s==1){
	  continue;
	}
	
	array_t<std::size_t> elec_ind("elec_ind");
	array_t<std::size_t> pos_ind("pos_ind");

	std::size_t elec_spec;
	elec_spec = 2;
	auto& electrons  = domain.species[elec_spec];
	auto offset_elec = electrons.npart();
	auto ux1_elec    = electrons.ux1;
	auto ux2_elec    = electrons.ux2;
	auto ux3_elec    = electrons.ux3;
	auto i1_elec     = electrons.i1;
	auto i2_elec     = electrons.i2;
	auto dx1_elec    = electrons.dx1;
	auto dx2_elec    = electrons.dx2;
	auto phi_elec    = electrons.phi;
	auto weight_elec = electrons.weight;
	auto tag_elec    = electrons.tag;
	auto pld_elec    = electrons.pld;

	std::size_t pos_spec;
	pos_spec = 3;
	auto& positrons = domain.species[pos_spec];
	auto offset_pos = positrons.npart();
	auto ux1_pos    = positrons.ux1;
	auto ux2_pos    = positrons.ux2;
	auto ux3_pos    = positrons.ux3;
	auto i1_pos     = positrons.i1;
	auto i2_pos     = positrons.i2;
	auto dx1_pos    = positrons.dx1;
	auto dx2_pos    = positrons.dx2;
	auto phi_pos    = positrons.phi;
	auto weight_pos = positrons.weight;
	auto tag_pos    = positrons.tag;
	auto pld_pos    = positrons.pld;

	auto& species = domain.species[s];
	auto ux1    = species.ux1;
	auto ux2    = species.ux2;
	auto ux3    = species.ux3;
	auto i1     = species.i1;
	auto i2     = species.i2;
	auto dx1    = species.dx1;
	auto dx2    = species.dx2;
	auto phi    = species.phi;
	auto weight = species.weight;
	auto tag    = species.tag;
	auto pld    = species.pld;

	const auto q_ovr_m = species.charge() / species.mass();
	
	Kokkos::parallel_for(
          "CurvatureCooling", species.rangeActiveParticles(), Lambda(index_t p) {
	    if (tag(p) == ParticleTag::dead) {
	      return;
	    }
              
	    auto px      = ux1(p);
	    auto py      = ux2(p);
	    auto pz      = ux3(p);
	    real_t gamma = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));
	    auto pld0    = pld(p,0);  
          
	    const auto   i { i1(p) + N_GHOSTS };
	    const real_t dx1_ { dx1(p) };

	    const auto   j { i2(p) + N_GHOSTS };
	    const real_t dx2_ { dx2(p) };   
	    
	    const coord_t<Dim::_2D> xc2d{static_cast<real_t>(i1(p)) + dx1(p),
					 static_cast<real_t>(i2(p)) + dx2(p)};               
	    coord_t<Dim::_2D> xPh { ZERO };
	    metric.template convert<Crd::Cd, Crd::Ph>(xc2d, xPh);
	      
	    auto beta_x = px/gamma;
	    auto beta_y = py/gamma;
	    auto beta_z = pz/gamma;
	    
	    //auto  rand_gen = random_pool.get_state();
	    //auto randNum = Random<real_t>(rand_gen);
	    //random_pool.free_state(rand_gen);

	    auto randNum = randf(p + 0);
	    
 	    real_t dipoleConst = xPh[0]/(pow(math::sin(xPh[1]),2)+1e-6);
	    real_t profile = 5+(PairDensity-5.0)*0.5*(math::tanh((dipoleConst-4.0)/0.02)+math::tanh((20.0-dipoleConst)/0.02));
	    real_t rhoMax = profile/pow(xPh[0],2) * (math::tanh((rad_radius-xPh[0])/0.25)+1.0) / 2.0;
	    real_t densityFactor = math::exp(-math::pow(BCKP(i, j, 0)/rhoMax,2));

	    auto gamma_thresEff = gamma_thres;
	    if (xPh[1]<PolarCap){
	      gamma_thresEff = gamma_thres /(math::pow(xPh[1]/PolarCap, 4.0/7.0) + 1e-6);
	    }else if (xPh[1] > (constant::PI-PolarCap)){
	      gamma_thresEff = gamma_thres /(math::pow((constant::PI-xPh[1])/PolarCap, 4.0/7.0) + 1e-6);
	    }
	      	    
            auto flag = 0;
            if ((gamma > gamma_thresEff) && (randNum < stepsEmit*densityFactor)){
	      flag = 1;
            }
	    
	    if (flag==1){
	      
	      auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
	      auto pos_p = Kokkos::atomic_fetch_add(&pos_ind(), 1);
	      
	      const vec_t<Dim::_3D> betaCart {beta_x, beta_y, beta_z};
	      vec_t<Dim::_3D> betaTetr;
	      metric.template transform<Idx::U, Idx::T>(xc2d, betaCart, betaTetr);
	      
	      auto mom = math::sqrt(gammaSec*gammaSec-1);
	      i1_elec(elec_p + offset_elec) = i1(p);
	      dx1_elec(elec_p + offset_elec) = dx1(p);
	      i2_elec(elec_p + offset_elec) = i2(p);
	      dx2_elec(elec_p + offset_elec) = dx2(p);
	      phi_elec(elec_p + offset_elec) = phi(p);
	      //auto rand_gen1 = random_pool.get_state();
	      //auto direction_factor = Random<real_t>(rand_gen1);
	      //random_pool.free_state(rand_gen1);
	      auto direction_factor = randf(p + 1);
	      direction_factor = direction_factor * 0.5*(math::tanh((dipoleConst-4.0)/0.02)+math::tanh((20.0-dipoleConst)/0.02));
 	      if ((betaTetr[0] < 0)  && (direction_factor < FlippingFraction) && (xPh[0]>(rad_radius-FlippingDistance))){ 
		ux1_elec(elec_p + offset_elec) = -1*mom * beta_x;
		ux2_elec(elec_p + offset_elec) = -1*mom * beta_y;
		ux3_elec(elec_p + offset_elec) = -1*mom * beta_z;
	      }else{
		ux1_elec(elec_p + offset_elec) = mom * beta_x;
		ux2_elec(elec_p + offset_elec) = mom * beta_y;
		ux3_elec(elec_p + offset_elec) = mom * beta_z;
	      }
	      weight_elec(elec_p + offset_elec) = weight(p);
	      tag_elec(elec_p + offset_elec) = ParticleTag::alive;
	      pld_elec(elec_p + offset_elec, 0) = 0.0;
	      
	      i1_pos(pos_p + offset_pos) = i1(p);
	      dx1_pos(pos_p + offset_pos) = dx1(p);
	      i2_pos(pos_p + offset_pos) = i2(p);
	      dx2_pos(pos_p + offset_pos) = dx2(p);
	      phi_pos(pos_p + offset_pos) = phi(p);
	      if ((betaTetr[0] < 0) && (direction_factor < FlippingFraction) && (xPh[0]>(rad_radius-FlippingDistance))){
		ux1_pos(pos_p + offset_pos) = -1*mom * beta_x;
		ux2_pos(pos_p + offset_pos) = -1*mom * beta_y;
		ux3_pos(pos_p + offset_pos) = -1*mom * beta_z;
	      }else{
		ux1_pos(pos_p + offset_pos) = mom * beta_x;
		ux2_pos(pos_p + offset_pos) = mom * beta_y;
		ux3_pos(pos_p + offset_pos) = mom * beta_z;		  
	      }
	      
	      weight_pos(pos_p + offset_pos) = weight(p);
	      tag_pos(pos_p + offset_pos) = ParticleTag::alive;
	      pld_pos(pos_p + offset_pos, 0) = 0.0;
	    }
	    //random_pool.free_state(rand_gen);
	  });
        auto elec_ind_h = Kokkos::create_mirror(elec_ind);
        Kokkos::deep_copy(elec_ind_h, elec_ind);
        electrons.set_npart(offset_elec + elec_ind_h());
        
        auto pos_ind_h = Kokkos::create_mirror(pos_ind);
        Kokkos::deep_copy(pos_ind_h, pos_ind);
        positrons.set_npart(offset_pos + pos_ind_h());
      }
    }

    void CustomFieldOutput(const std::string&   name,
                           ndfield_t<M::Dim, 6> buffer,
                           index_t              index,
                           timestep_t,
                           simtime_t,
                           const Domain<S, M>&  domain) {
      if (name == "density") {
        if constexpr (M::Dim == Dim::_2D) {
          const auto& EM = domain.fields.em;
          auto metric = domain.mesh.metric;
          const auto use_weights = M::CoordType != Coord::Cart;
          const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");

          for (std::size_t s { 0 }; s < 4; ++s) {
            auto& species = domain.species[s];
            auto i1     = species.i1;
            auto i2     = species.i2;
            auto weight = species.weight;
            auto tag    = species.tag;
            const auto m = species.mass();

            Kokkos::parallel_for(
                 "density", species.rangeActiveParticles(), Lambda(index_t p) {
                   if (tag(p) == ParticleTag::dead) {
                     return;
                   }

                   auto coeff = inv_n0*m /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF });
                   if (use_weights) {
                     coeff *= weight(p);
                   }
                   buffer(i1(p)+N_GHOSTS, i2(p)+N_GHOSTS, index) +=coeff;
                 });
          }
        }
      }else if (name == "parallelCurrent"){
	if constexpr (M::Dim == Dim::_2D) {
	    const auto& EB = domain.fields.em;
	    auto metric = domain.mesh.metric;
	    Kokkos::parallel_for("ParCurr",
		 domain.mesh.rangeActiveCells(),
		 Lambda(index_t i1, index_t i2) {
		   const real_t       i1_ { COORD(i1) };
		   const real_t       i2_ { COORD(i2) };
	    
		   const real_t inv_sqrt_detH_0pH { ONE /
						    metric.sqrt_det_h({ i1_, i2_ + HALF }) };
		   const real_t h3_mHpH { metric.template h_<3, 3>({ i1_ - HALF, i2_ + HALF }) };
		   const real_t h3_pHpH { metric.template h_<3, 3>({ i1_ + HALF, i2_ + HALF }) };
		   const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
		   const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h({ i1_ + HALF, i2_ }) };
		   const real_t h1_0mH { metric.template h_<1, 1>({ i1_, i2_ - HALF }) };
		   const real_t h1_0pH { metric.template h_<1, 1>({ i1_, i2_ + HALF }) };
		   const real_t h2_pH0 { metric.template h_<2, 2>({ i1_ + HALF, i2_ }) };
		   const real_t h2_mH0 { metric.template h_<2, 2>({ i1_ - HALF, i2_ }) };
		   const real_t h3_pHmH { metric.template h_<3, 3>({ i1_ + HALF, i2_ - HALF }) };
		   const real_t curlBdotB = inv_sqrt_detH_pH0 *(h3_pHpH * EB(i1, i2, em::bx3) -
								h3_pHmH * EB(i1, i2 - 1, em::bx3))*INV_4*
		     (EB(i1, i2, em::bx1)+EB(i1+1, i2, em::bx1)+EB(i1, i2-1, em::bx1)+EB(i1+1, i2-1, em::bx1)) + 
		     inv_sqrt_detH_0pH *(h3_mHpH * EB(i1 - 1, i2, em::bx3) - h3_pHpH * EB(i1, i2, em::bx3))*INV_4*
		     (EB(i1, i2, em::bx2)+EB(i1-1, i2, em::bx2)+EB(i1, i2+1, em::bx2)+EB(i1-1, i2+1, em::bx2)) +
		     inv_sqrt_detH_00 * (h1_0mH * EB(i1, i2 - 1, em::bx1) - h1_0pH * EB(i1, i2, em::bx1) +
					 h2_pH0 * EB(i1, i2, em::bx2) - h2_mH0 * EB(i1 - 1, i2, em::bx2))*INV_4
		     *(EB(i1, i2, em::bx3)+EB(i1-1, i2, em::bx3)+EB(i1, i2-1, em::bx3)+EB(i1-1, i2-1, em::bx3));
		   const real_t B2 = EB(i1, i2, em::bx1)*EB(i1, i2, em::bx1)+EB(i1, i2, em::bx2)*EB(i1, i2, em::bx2)+
		     EB(i1, i2, em::bx3)*EB(i1, i2, em::bx3);
		   buffer(i1, i2, index) = curlBdotB/B2;
		 });
	  }
      }else if (name == "strongCooling"){
	if constexpr (M::Dim == Dim::_2D) {
	    const auto& EB = domain.fields.em;
	    auto metric = domain.mesh.metric;
	    const auto use_weights = M::CoordType != Coord::Cart;
	    const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");

	    for (std::size_t s { 0 }; s < 4; ++s) {

	      if(s==1){
		continue;
	      }
	      
	      auto& species = domain.species[s];
	      auto i1     = species.i1;
	      auto i2     = species.i2;
	      auto dx1    = species.dx1;
	      auto dx2    = species.dx2;
	      auto phi    = species.phi;	      
	      auto weight = species.weight;
	      auto tag    = species.tag;
	      auto ux1    = species.ux1;
	      auto ux2    = species.ux2;
	      auto ux3    = species.ux3;	
	      const auto m = species.mass();

	      const auto sync_grad = params.template
		get<real_t>("algorithms.synchrotron.gamma_rad");
	      const auto coeff_sync = (real_t)(0.1) * dt *
		params.template get<real_t>("scales.omegaB0")/
		(SQR(sync_grad) * species.mass());
	      
	      Kokkos::parallel_for(
                 "density", species.rangeActiveParticles(), Lambda(index_t p) {
                   if (tag(p) == ParticleTag::dead) {
                     return;
                   }

		   vec_t<Dim::_3D> e0 { ZERO }, b0 { ZERO };
                   const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1(p)) + dx1(p),
                                            static_cast<real_t>(i2(p)) + dx2(p), phi(p)};

                   const auto   i { i1(p) + N_GHOSTS };
                   const real_t dx1_ { dx1(p) };

                   const auto   j { i2(p) + N_GHOSTS };
                   const real_t dx2_ { dx2(p) };

                   vec_t<Dim::_3D> b_int { ZERO };
                   vec_t<Dim::_3D> e_int { ZERO };

                   real_t      c000, c100, c010, c110, c00, c10;

                   c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
                   c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
                   c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
                   c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));

                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;


                   c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
                   c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
                   c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
                   c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
                               EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
                   c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
                               EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
                   c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
                               EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
                   c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
                               EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;

                   metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, b_int, b0);

                   c000  = HALF * (EB(i, j, em::ex1) + EB(i - 1, j, em::ex1));
                   c100  = HALF * (EB(i, j, em::ex1) + EB(i + 1, j, em::ex1));
                   c010  = HALF * (EB(i, j + 1, em::ex1) + EB(i - 1, j + 1, em::ex1));
                   c110  = HALF * (EB(i, j + 1, em::ex1) + EB(i + 1, j + 1, em::ex1));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = HALF * (EB(i, j, em::ex2) + EB(i, j - 1, em::ex2));
                   c100  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j - 1, em::ex2));
                   c010  = HALF * (EB(i, j, em::ex2) + EB(i, j + 1, em::ex2));
                   c110  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j + 1, em::ex2));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = EB(i, j, em::ex3);
                   c100  = EB(i + 1, j, em::ex3);
                   c010  = EB(i, j + 1, em::ex3);
                   c110  = EB(i + 1, j + 1, em::ex3);
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;

                   metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, e_int, e0);
		   
		   vec_t<Dim::_3D> u_prime {ux1(p), ux2(p), ux3(p)};

		   real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
		   u_prime[0]             *= gamma_prime_sqr;
		   u_prime[1]             *= gamma_prime_sqr;
		   u_prime[2]             *= gamma_prime_sqr;
		   gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);
		   const real_t beta_dot_e {
		       DOT(u_prime[0], u_prime[1], u_prime[2], e0[0], e0[1], e0[2])
		   };
		   vec_t<Dim::_3D> e_plus_beta_cross_b {
                        e0[0] + CROSS_x1(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
                        e0[1] + CROSS_x2(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
                        e0[2] + CROSS_x3(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2])
		   };
		   const real_t chiR_sqr { NORM_SQR(e_plus_beta_cross_b[0],
						    e_plus_beta_cross_b[1],
						    e_plus_beta_cross_b[2]) -
					   SQR(beta_dot_e) };
		   const real_t loss = coeff_sync * gamma_prime_sqr * chiR_sqr;
		   if (loss > 0.2 * math::sqrt(gamma_prime_sqr)){

		     auto coeff = inv_n0*m /
		       metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
					   static_cast<real_t>(i2(p)) + HALF });
		     if (use_weights) {
		       coeff *= weight(p);
		     }
		     buffer(i1(p)+N_GHOSTS, i2(p)+N_GHOSTS, index) +=coeff;
		   }
                 });
	    }
	  }
      }else if (name == "densityGCA"){
        if constexpr (M::Dim == Dim::_2D) {
          const auto& EB = domain.fields.em;
          auto metric = domain.mesh.metric;

          const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");
          const auto gcaLarmor {params.template get<real_t>("algorithms.gca.larmor_max")};
          const auto gcaEoverB_ {params.template get<real_t>("algorithms.gca.e_ovr_b_max")};
          const auto gcaEoverB {SQR(gcaEoverB_)};
          const auto larm {params.template get<real_t>("scales.larmor0")};
          const auto use_weights = M::CoordType != Coord::Cart;

          for (std::size_t s { 0 }; s < 4; ++s) {
            auto& species = domain.species[s];
            auto ux1    = species.ux1;
            auto ux2    = species.ux2;
            auto ux3    = species.ux3;
            auto i1     = species.i1;
            auto i2     = species.i2;
            auto dx1    = species.dx1;
            auto dx2    = species.dx2;
            auto phi    = species.phi;
            auto weight = species.weight;
            auto tag    = species.tag;
            const auto m = species.mass();
            const auto q_ovr_m = species.charge() / species.mass();

            Kokkos::parallel_for(
                 "GCA_density", species.rangeActiveParticles(), Lambda(index_t p) {

                   if (tag(p) == ParticleTag::dead) {
                     return;
                   }

                   auto gamma = math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)));

                   const coord_t<Dim::_3D> xc3d {static_cast<real_t>(i1(p)) + dx1(p),
                                            static_cast<real_t>(i2(p)) + dx2(p), phi(p)};

                   vec_t<Dim::_3D> b_int_Cart { ZERO };
                   vec_t<Dim::_3D> e_int_Cart { ZERO };

                   const auto   i { i1(p) + N_GHOSTS };
                   const real_t dx1_ { dx1(p) };

                   const auto   j { i2(p) + N_GHOSTS };
                   const real_t dx2_ { dx2(p) };

                   vec_t<Dim::_3D> b_int { ZERO };
                   vec_t<Dim::_3D> e_int { ZERO };

                   real_t      c000, c100, c010, c110, c00, c10;

                   c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
                   c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
                   c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
                   c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));

                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;


                   c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
                   c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
                   c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
                   c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
                               EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
                   c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
                               EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
                   c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
                               EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
                   c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
                               EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   b_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;

                   metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, b_int, b_int_Cart);

                   c000  = HALF * (EB(i, j, em::ex1) + EB(i - 1, j, em::ex1));
                   c100  = HALF * (EB(i, j, em::ex1) + EB(i + 1, j, em::ex1));
                   c010  = HALF * (EB(i, j + 1, em::ex1) + EB(i - 1, j + 1, em::ex1));
                   c110  = HALF * (EB(i, j + 1, em::ex1) + EB(i + 1, j + 1, em::ex1));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[0] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = HALF * (EB(i, j, em::ex2) + EB(i, j - 1, em::ex2));
                   c100  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j - 1, em::ex2));
                   c010  = HALF * (EB(i, j, em::ex2) + EB(i, j + 1, em::ex2));
                   c110  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j + 1, em::ex2));
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[1] = c00 * (ONE - dx2_) + c10 * dx2_;

                   c000  = EB(i, j, em::ex3);
                   c100  = EB(i + 1, j, em::ex3);
                   c010  = EB(i, j + 1, em::ex3);
                   c110  = EB(i + 1, j + 1, em::ex3);
                   c00   = c000 * (ONE - dx1_) + c100 * dx1_;
                   c10   = c010 * (ONE - dx1_) + c110 * dx1_;
                   e_int[2] = c00 * (ONE - dx2_) + c10 * dx2_;

                   metric.template transform_xyz<Idx::U, Idx::XYZ>(xc3d, e_int, e_int_Cart);

                   const auto E2 { NORM_SQR(e_int_Cart[0], e_int_Cart[1], e_int_Cart[2]) };
                   const auto B2 { NORM_SQR(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
                   const auto rL { gamma * larm/ (math::abs(q_ovr_m) * math::sqrt(B2)) };

                   if (B2 > ZERO && rL < gcaLarmor && (E2 / B2) < gcaEoverB ) {
                     auto coeff = inv_n0 * m /
                       metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                           static_cast<real_t>(i2(p)) + HALF });
                     if (use_weights) {
                       coeff *= weight(p);
                     }
                     buffer(i1(p)+N_GHOSTS, i2(p)+N_GHOSTS, index) +=coeff;
                   }
                 });
          }
        }
      }else {
        raise::Error("Custom output not provided", HERE);
      }
    }    
  };
  
} // namespace user

#endif
