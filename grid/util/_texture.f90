!  ----------------------------------------------------------------------------
!  grid.util._texture
!  ==================
!
!  Fortran submodule for computing texture fields from gridded radar data.
!
!  ----------------------------------------------------------------------------


subroutine compute_texture(field, x_window, y_window, fill_value, &
                           nx, ny, nz, sigma, sample)
!  ----------------------------------------------------------------------------
!
!  Parameters
!  ----------
!  field : array, dim(z,y,x), float64
!
!  x_window : int32
!
!  y_window : int32
!
!  fill_value : float64
!
!  Returns
!  -------
!  sigma, array, dim(z,y,x), float64
!
!  sample : array, dim(z,y,x), int32
!     Sample size within the texture window. Note that the minimum sample size
!     within the texture window needed to compute the texture field is 2 since
!     the unbiased estimator of variance is used.
!
!  ----------------------------------------------------------------------------

   implicit none

   integer(kind=4), intent(in)                       :: nz, ny, nx
   real(kind=8), intent(in), dimension(nz,ny,nx)     :: field
   integer(kind=4), intent(in)                       :: x_window, y_window
   real(kind=8), intent(in)                          :: fill_value
   real(kind=8), intent(out), dimension(nz,ny,nx)    :: sigma
   integer(kind=4), intent(out), dimension(nz,ny,nx) :: sample

!  Define local variables
   logical, dimension(nz,ny,nx) :: mask
   real(kind=8), parameter      :: atol=1.e-5
   real(kind=8)                 :: mean, var
   integer(kind=4)              :: N, i, j, k, x0, xf, y0, yf


!  F2PY directives
!  f2py integer(kind=4), optional, intent(in) :: nx, ny, nz
!  f2py real(kind=8), intent(in)              :: field, fill_value
!  f2py integer(kind=4), intent(in)           :: x_window, y_window
!  f2py real(kind=8), intent(out)             :: sigma
!  f2py integer(kind=4), intent(out)          :: sample

   sigma = fill_value
   mask = abs(field - fill_value) > atol

   do i = 1, nx

!  Determine stencil of x grid points in texture window
   x0 = max(1, i - x_window / 2)
   xf = min(nx, i + x_window / 2)

   do j = 1, ny

!  Determine stencil of y grid points in texture window
   y0 = max(1, j - y_window / 2)
   yf = min(ny, j + y_window / 2)

   do k = 1, nz

      if (mask(k,j,i)) then

!     Compute the sample size within the texture window
      N = count(mask(k,y0:yf,x0:xf))
      sample(k,j,i) = N

!
      if (N > 1) then

      mean = sum(field(k,y0:yf,x0:xf), mask(k,y0:yf,x0:xf)) / N
      var = sum((field(k,y0:yf,x0:xf) - mean)**2, mask(k,y0:yf,x0:xf)) / (N - 1)
      sigma(k,j,i) = sqrt(var)

      endif
      endif

   enddo
   enddo
   enddo

   return

end subroutine compute_texture
