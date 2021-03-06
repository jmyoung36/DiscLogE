/*
c_calculateCoefficientsCrossExp.c

simple C to calculate a matrix of coefficients for the gradients in DiscLogECoeffExp
do this in place to a provided array of coeffients, until I figure out how to return a numpy array!

*/

void c_calculateCoefficientsCrossExp (double* x_logs, double* z_logs, double* x_derivatives, double* z_derivatives, double* x_log_eigenvals, double* z_log_eigenvals, double* coefficients, int n_x, int n_z, int dd, int d) {

    /* decalare loop variables */    
    int i, j, k, l ;
    
    /* declare subject log eigenvalues */
    double x_log_eigenval, z_log_eigenval = 0;

    /* loop through all pairs.
    Take advantage of pairwise symmetry to only traverse lower triangle */
    for (i = 0; i < n_x; i++) {
            
        // log eigenvalue for this subject
        x_log_eigenval = x_log_eigenvals[i];    
            
            
        for (j = 0; j < n_z; j++) {
                
            // log eigenvalue for this subject
            z_log_eigenval = z_log_eigenvals[j];
                
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
                    lo_tri_sum = lo_tri_sum + ( (x_logs[index+i*dd] - z_logs[index+j*dd]) * ((x_derivatives[index+i*dd] * x_log_eigenval) - (z_derivatives[index+j*dd] * z_log_eigenval)) ); 
                    }
                
                /* index of diagonal element */
                int index = k + k*d;
                diag_sum = diag_sum + ( (x_logs[index+i*dd] - z_logs[index+j*dd]) * ((x_derivatives[index+i*dd] * x_log_eigenval) - (z_derivatives[index+j*dd] * z_log_eigenval) ));
                }           
                     
            /* store the element value in coefficients array
            remembering to double sum of lower triangular elements */
            coefficients[j+i*n_z] = diag_sum + 2*lo_tri_sum;
            }
        }
    return;
}
