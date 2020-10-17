A simple cpp header for calling plot functions in python. It can work with vector type data with index operator [] is defined, such as std::vector, Eigen::VectorXd, etc.

Requirement:

    1. python3.x installed.
    2. make sure "python\Python.h" in your include path. For example, you can do
        
            ln -s yourPythonRoot/Headers /usr/local/include/python

       or just simply change "#include <python/Python.h>" in head file to #include "your Python.h"
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
    }

Remark:
    Slowly developing, the goal is to wrap some basic functions in python matplotlib, and still keep its simple syntax. 