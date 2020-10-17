A simple cpp header for calling plot functions in python. It can work with vector type data with index operator [] is defined, such as std::vector, Eigen::VectorXd, etc.

Requirement:

    1. python3.x installed.
    2. make sure Python.h in your include path.
    3. make sure libpython3.x (or equivalent one, incase you change this lib to your custom name), libpython2.* does not work!

Install: put the header into your dir.

Example:

    #include "matplot.h"
    using namespace plt;

    int main() 
    { 
        /*
            x, y are your vectors with same length or just basic type(int, double)
        */
        figure();           // like plt.figure() in python
        plot(x, y, "o-");   // like plt.plot(x, y, 'o-') in python, here " is necessary, ' does not work
        show();             // like plt.show() in python
    }

Remark:
    Slowly developing, the goal is to wrap some basic functions in python matplotlib, and still keep its simple syntax. 