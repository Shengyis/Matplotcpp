A simple cpp header for calling plot functions in python. It can work with vector type data with index operator [] is defined, such as std::vector, Eigen::VectorXd, etc.

Requirement:

    1. python3.x installed.
    2. make sure "Python.h" in your include path. 
    3. when compiling, add -lpython3.x (or equivalent one, incase you change this lib to your custom name), -lpython2.x does not work!

Install: put the header into your dir.

Example:

    #include "matplot.h"
    using namespace plt;

    int main() 
    { 
        /*
            x, y are your vectors with same length or just basic type(like int, double)
        */
        figure();                                   // like plt.figure() in python
        plot(x, y, "o-");                           // like plt.plot(x, y, 'o-') in python, here " is necessary, ' does not work
        plot(x, y, ".-", {{"label", "stars"}})      // like plt.plot(x, y, 'o', label = 'stars') in python
        legend()                                    // like plt.legend() in python
        show();                                     // like plt.show() in python

        /*
            x, y are Eigen::VectorXd, the followings are 2d contour/contourf plot
        */
        Eigen::MatrixXd X, Y, Z;                    
        tie(X, Y) = meshgrid(x, y);                 // meshgrid function only works for Eigen::VectorXd
        Z = your_function(X, Y);
        contourf(X, Y, Z, 100);                     // default only plot 10 levels, 
        colorbar();                                 // another option is plotting specific level curves 
        show();                                     // such as contour(X, Y, Z, {20.0, 30.0, 40.5})
                                                    // or define vector<double> / Eigen::VectorXd / std::initializer_list<double> levels, 
                                                    // then contour(X, Y, Z, levels);

    }

Remark:
    Slowly developing, the goal is to wrap some basic functions in python matplotlib, and still keep its simple syntax. 