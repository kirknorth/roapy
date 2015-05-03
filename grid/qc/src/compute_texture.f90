! Module: compute_texture

subroutine compute(input, x_window, y_window, fill_value, nz, ny, nx, &
                   sample, texture)

   implicit none

   integer(kind=4), intent(in)                       :: nz, ny, nx
   integer(kind=4), intent(in)                       :: x_window, y_window
   real(kind=8), intent(in)                          :: fill_value
   real(kind=8), intent(in), dimension(nz,ny,nx)     :: input
   integer(kind=4), intent(out), dimension(nz,ny,nx) :: sample
   real(kind=8), intent(out), dimension(nz,ny,nx)    :: texture

!  Define local variables =====================================================

   logical, dimension(nz,ny,nx) :: is_valid
   integer(kind=4)            :: N_tmp
   real(kind=8)               :: mean_tmp, var_tmp
   integer(kind=4)            :: i, j, k, x0, xf, y0, yf

!  ============================================================================

!  Fill texture array
   texture = fill_value

!  Define valid grid points
   is_valid = input /= fill_value

   do i = 1, nx

!     Determine stencil of x grid points in window
      x0 = max(1, i - x_window / 2)
      xf = min(nx, i + x_window / 2)

      do j = 1, ny

!        Determine stencil of y grid points in window
         y0 = max(1, j - y_window / 2)
         yf = min(ny, j + y_window / 2)

         do k = 1, nz

!           Compute the sample size within the 2-D window
            N_tmp = count(is_valid(k,y0:yf,x0:xf))
            sample(k,j,i) = N_tmp

!           Only compute the texture field if the sample size within the 2-D
!           window is greater than zero
            if (N_tmp > 0) then

            mean_tmp = sum(input(k,y0:yf,x0:xf), is_valid(k,y0:yf,x0:xf)) / N_tmp
            var_tmp = sum(input(k,y0:yf,x0:xf)**2, is_valid(k,y0:yf,x0:xf)) / N_tmp - mean_tmp**2
            texture(k,j,i) = sqrt(var_tmp)

            endif

         enddo
      enddo
   enddo

   return

end subroutine compute
