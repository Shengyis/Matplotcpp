A simple cpp header for calling plot functions in python. Now it can work directly with dense vectors in Eigen3.

Requirement:
    1. python3.x, Eigen3 for vectors plotting.
    2. make sure Python.h in your include path.
    3. make sure libpython3.8 (or equivalent one, incase you change this lib to your custom name), libpython2.* does not work!

Install: put the header into your dir.

Example:
    #include "matplot.h">
    using namespace plt;
    /*
        x, y are your eigen vector
    */

    figure();           // like plt.figure() in python
    plot(x, y, "o-");   // like plt.plot(x, y, 'o-') in python, here " is necessary, ' does not working
    show();             // like plt.show() in python

Remark:
    Slowly developing, the goal is to wrap some basic functions in python matplotlib, and still keep its simple syntax. 