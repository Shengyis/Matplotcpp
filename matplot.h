#pragma once
#include <eigen3/Eigen/Core>
#include <python/Python.h>

namespace plt
{
    static bool does_py_init = 0;
    static bool does_py_stop = 0;
    static bool does_load_pyplot = 0;

    static PyObject *pModule;

    void py_init()
    {
        if (!does_py_init)
        {
            Py_Initialize();
            does_py_init = 1;
        }
    }

    void load_pyplot()
    {
        py_init();
        if (!does_load_pyplot)
        {
            PyRun_SimpleString("import matplotlib");
            PyRun_SimpleString("matplotlib.rcParams['text.usetex'] = True");
            pModule = PyImport_ImportModule("matplotlib.pyplot");

            does_load_pyplot = 1;
        }
    }

    PyObject *getPltFun(const char *argv)
    {
        return PyObject_GetAttrString(pModule, argv);
    }

    void setVal(PyObject *pV, const Eigen::VectorXd &v)
    {
        int n = PyList_Size(pV);
        for (int i = 0; i < n; ++i)
            PyList_SetItem(pV, i, PyFloat_FromDouble(v(i)));
    }

    void figure()
    {
        load_pyplot();
        PyObject_CallFunctionObjArgs(getPltFun("figure"), NULL);
    }

    void show()
    {
        PyObject_CallFunctionObjArgs(getPltFun("show"), NULL);
    }

    void plot(const double &x, const double &y)
    {
        PyObject_CallFunctionObjArgs(getPltFun("plot"), PyFloat_FromDouble(x), PyFloat_FromDouble(y), NULL);
    }

    void plot(const double &x, const double &y, const char *argv)
    {
        PyObject_CallFunctionObjArgs(getPltFun("plot"), PyFloat_FromDouble(x), PyFloat_FromDouble(y), PyUnicode_FromString(argv), NULL);
    }

    void plot(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
    {
        int n = x.size();
        static PyObject *px = PyList_New(n);
        static PyObject *py = PyList_New(n);
        setVal(px, x);
        setVal(py, y);
        PyObject_CallFunctionObjArgs(getPltFun("plot"), px, py, NULL);
    }

    void plot(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const char *argv)
    {
        int n = x.size();
        static PyObject *px = PyList_New(n);
        static PyObject *py = PyList_New(n);
        setVal(px, x);
        setVal(py, y);
        PyObject_CallFunctionObjArgs(getPltFun("plot"), px, py, PyUnicode_FromString(argv), NULL);
    }

    void xlim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void ylim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void title(const char *argv)
    {
        PyObject_CallFunctionObjArgs(getPltFun("title"), PyUnicode_FromString(argv), NULL);
    }

    void xlabel(const char *argv)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlabel"), PyUnicode_FromString(argv), NULL);
    }

    void ylabel(const char *argv)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylabel"), PyUnicode_FromString(argv), NULL);
    }
} // namespace plt