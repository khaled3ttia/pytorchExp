

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 2) #2
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %7, %3
"i32B

	full_text


i32 %7
4icmpB,
*
	full_text

%13 = icmp slt i32 %9, %2
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%14 = and i1 %12, %13
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %11, %1
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%16 = and i1 %14, %15
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %15
8brB2
0
	full_text#
!
br i1 %16, label %17, label %25
!i1B

	full_text


i1 %16
4mul8B+
)
	full_text

%18 = mul nsw i32 %7, %2
$i328B

	full_text


i32 %7
1add8B(
&
	full_text

%19 = add i32 %18, %9
%i328B

	full_text
	
i32 %18
$i328B

	full_text


i32 %9
1mul8B(
&
	full_text

%20 = mul i32 %19, %1
%i328B

	full_text
	
i32 %19
1add8B(
&
	full_text

%21 = add i32 %11, %4
%i328B

	full_text
	
i32 %11
2add8B)
'
	full_text

%22 = add i32 %21, %20
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %20
6sext8B,
*
	full_text

%23 = sext i32 %22 to i64
%i328B

	full_text
	
i32 %22
^getelementptr8BK
I
	full_text<
:
8%24 = getelementptr inbounds double, double* %0, i64 %23
%i648B

	full_text
	
i64 %23
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %24, align 8, !tbaa !8
-double*8B

	full_text

double* %24
'br8B

	full_text

br label %25
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %1
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
4double8B&
$
	full_text

double 0.000000e+00
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1       	  
 

                      !" !# !! $% $$ &' && () (( *, 
- - . &/ 0 0    	  
             " #! %$ '& )  +* + + 11 11  11  11 2 (3 4 5 "
kernel_zran3_3"
_Z13get_global_idj*?
npb-MG-kernel_zran3_3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize
@

wgsize_log1p
???A

transfer_bytes	
????
 
transfer_bytes_log1p
???A

devmap_label
