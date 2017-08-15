/*
c_calculateCoefficientsTrainExp.c

simple C to calculate a matrix of coefficients for the gradients in DiscLogECoeffExp
do this in place to a provided array of coeffients, until I figure out how to return a numpy array!

*/

void c_calculateCoefficientsTrainExpPointer (double* x_logs, double* derivatives, double* x_log_eigenvals, double* coefficients, int n, int dd, int d) {

    /* decalare loop variables */    
    int i, j, k, l ;
    
    /* declare subject log eigenvalues */
    double x_log_eigenval_i, x_log_eigenval_j = 0;
    
    /* declare pointer variables to element of the x_log_eigenvals array */
    double *ptr_i; double *ptr_j;

    /* loop through all pairs.
    Take advantage of pairwise symmetry to only traverse lower triangle */
    for (i = 0; i < n; i++) {
        
        // log eigenvalue for this subject
        ptr_i = x_log_eigenvals + (i * d);
        x_log_eigenval_i = *ptr_i;                            
            
        for (j = 0; j < i; j++) {
                
            // log eigenvalue for this subject
            ptr_j = x_log_eigenvals + (j * d);
            x_log_eigenval_j = *ptr_j;
                
            /* initialise elements
            one to sum lower triangle
            one to sum diagonal*/           
            double lo_tri_sum = 0;
            double diag_sum = 0;          

            /* again exploit symmetry for a speedup - theses vectors are squashed
            symmetric matrices, so only need to traverse lower triangle and reflect */
            for(k=0;k<d;k++) {
                for(l=0;l<k;l++) {
                        
                    /* index of lower triangular element */    
                    int index = l + k*d;
                    lo_tri_sum = lo_tri_sum + ( (x_logs[index+i*dd] - x_logs[index+j*dd]) * ( (derivatives[index+i*dd] * x_log_eigenval_i) - (derivatives[index+j*dd] * x_log_eigenval_j))); 
                    }
                
                /* index of diagonal element */
                int index = k + k*d;
                diag_sum = diag_sum + ( (x_logs[index+i*dd] - x_logs[index+j*dd]) * ( (derivatives[index+i*dd] * x_log_eigenval_i) - (derivatives[index+j*dd] * x_log_eigenval_j)));
                }           
                     
            /* store the element value in coefficients array
            remembering to double sum of lower triangular elements */
            coefficients[i+j*n] = diag_sum + 2*lo_tri_sum;
            coefficients[j+i*n] = diag_sum + 2*lo_tri_sum;

            }
        }
    return;
}
