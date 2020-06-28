#include <cmath>
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <dlib/global_optimization.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ph = std::placeholders;
namespace py = pybind11;


double rice_pdf(double x, double nu, double sigma) {
    double sigmasq = sigma*sigma;
    return (x/sigmasq) * exp(-(x*x+nu*nu)/(2*sigmasq)) * std::cyl_bessel_i(0.0, x*nu/sigmasq);
}

double beta_pdf(double x, double alpha, double beta) {
    return pow(x, alpha-1) * pow(1-x, beta-1) / std::beta(alpha, beta);
}

double alpha_par(double m, double mod_m) {
    return ((1 - m) / (m * mod_m * mod_m) - 1) * m;
}

double beta_par(double m, double mod_m) {
    return ((1 - m) / (m * mod_m * mod_m) - 1) * (1 - m);
}

double P(double m_i, double m_obs, double m_0, double mod_m, double sigma_m) {
    double alpha_loc = alpha_par(m_0, mod_m);
    double beta_loc = beta_par(m_0, mod_m);
    return beta_pdf(m_i, alpha_loc, beta_loc) * rice_pdf(m_obs, m_i, sigma_m);
}

class Pint {
public:
    Pint(double m_obs, double m_0, double mod_m, double sigma_m) {
        m_obs_ = m_obs;
        m_0_ = m_0;
        mod_m_ = mod_m;
        sigma_m_ = sigma_m;
    }
    double m_obs_;
    double m_0_;
    double mod_m_;
    double sigma_m_;
    void operator() (const double &x, double &dxdt, const double t) {
        dxdt = P(t, m_obs_, m_0_, mod_m_, sigma_m_);
    }
};


double lik_j(double m_obs_j, double m_0, double m_mod, double sigma_j) {
    Pint pint(m_obs_j, m_0, m_mod, sigma_j);
    double pint_state = 0.0;
    using namespace boost::numeric::odeint;
    typedef runge_kutta_dopri5<double> stepper_type;

    int num_steps = integrate_adaptive(make_controlled(1E-10, 1E-10, 0.001, stepper_type()),
                                       pint,
                                       pint_state, 0.000001, 0.999999, 0.001);
    return pint_state;
}


double loglik(double m_0, double mod_m, std::vector<double> ms_obs, std::vector<double> sigmas) {
    double result = 0.0;
    for(size_t i=0;i<ms_obs.size();i++) {
        result += log(lik_j(ms_obs[i], m_0, mod_m, sigmas[i]));
    }
    if(isinf(result)) {
        return -result;
    }
    return result;
}

std::pair<double,double> fit(std::vector<double> ms_obs, std::vector<double> sigmas) {
    auto dlib_lik = [&ms_obs, &sigmas](double m_0, double mod_m_0) {
        double result = loglik(m_0, mod_m_0, ms_obs, sigmas);
        return result;
    };

    dlib::thread_pool tp(3);
    dlib::function_evaluation res = dlib::find_max_global(tp,
                                     dlib_lik,
                                     {0.001,0.001},
                                     {0.8,0.8},
                                     dlib::max_function_calls(1000)
    );
    return std::make_pair(res.x, res.y);
}


// Compile with:
//c++ -O3 -Wall -shared -std=c++17 -fPIC -ldlib -march=native -DNDEBUG -O3 -fext-numeric-literals `python3 -m pybind11 --includes` -o pypolinf`python3-config --extension-suffix` pypolinf.cpp
PYBIND11_MODULE(pypolinf, m) {
    using namespace pybind11::literals; // for _a literal to define arguments
    m.doc() = "Finding true mean intrinsic FPOL and intrinsic modulation index from sequence of the observed FPOL values and their errors."; // optional module docstring

    m.def("fit", &fit, "Returns true mean intrinsic FPOL and intrinsic modulation index",
          "Iterable of the observed FPOL values"_a,
          "Iterable of the errors of the observed FPOL values"_a);
}

