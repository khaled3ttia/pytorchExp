

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
3icmpB+
)
	full_text

%7 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
6brB0
.
	full_text!

br i1 %7, label %8, label %16
 i1B

	full_text	

i1 %7
/shl8B&
$
	full_text

%9 = shl i64 %5, 32
$i648B

	full_text


i64 %5
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
ygetelementptr8Bf
d
	full_textW
U
S%11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %10, i32 0
%i648B

	full_text
	
i64 %10
ygetelementptr8Bf
d
	full_textW
U
S%12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %10, i32 0
%i648B

	full_text
	
i64 %10
@bitcast8B3
1
	full_text$
"
 %13 = bitcast double* %11 to i8*
-double*8B

	full_text

double* %11
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %13, i8 0, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %13
^getelementptr8BK
I
	full_text<
:
8%14 = getelementptr inbounds double, double* %2, i64 %10
%i648B

	full_text
	
i64 %10
@bitcast8B3
1
	full_text$
"
 %15 = bitcast double* %12 to i8*
-double*8B

	full_text

double* %12
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %15, i8 0, i64 16, i1 false)
%i8*8B

	full_text
	
i8* %15
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %14, align 8, !tbaa !8
-double*8B

	full_text

double* %14
'br8B

	full_text

br label %16
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
6struct*8B'
%
	full_text

%struct.dcomplex* %1
6struct*8B'
%
	full_text

%struct.dcomplex* %0
,double*8B

	full_text


double* %2
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 0.000000e+00
%i18B

	full_text


i1 false
!i88B

	full_text

i8 0
$i648B

	full_text


i64 16
#i328B

	full_text	

i32 0      	  
 

                     !     	 
 
   
         "" ## ##  ""  ## $ $ 
% & & ' ' ( ( ) ) ) "	
init_ui"
_Z13get_global_idj"
llvm.memset.p0i8.i64*?
npb-FT-init_ui.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
@

devmap_label

 
transfer_bytes_log1p
@s?A

wgsize_log1p
@s?A

transfer_bytes
???