

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 1) #3
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
4icmpB,
*
	full_text

%10 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
5truncB,
*
	full_text

%11 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
5icmpB-
+
	full_text

%12 = icmp slt i32 %11, %2
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%13 = and i1 %10, %12
!i1B

	full_text


i1 %10
!i1B

	full_text


i1 %12
4icmpB,
*
	full_text

%14 = icmp slt i32 %9, %1
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%15 = and i1 %13, %14
!i1B

	full_text


i1 %13
!i1B

	full_text


i1 %14
8brB2
0
	full_text#
!
br i1 %15, label %16, label %30
!i1B

	full_text


i1 %15
0shl8B'
%
	full_text

%17 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%18 = ashr exact i64 %17, 32
%i648B

	full_text
	
i64 %17
0shl8B'
%
	full_text

%19 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%20 = ashr exact i64 %19, 32
%i648B

	full_text
	
i64 %19
0shl8B'
%
	full_text

%21 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
8mul8B/
-
	full_text 

%23 = mul nsw i64 %18, 21125
%i648B

	full_text
	
i64 %18
6mul8B-
+
	full_text

%24 = mul nsw i64 %20, 325
%i648B

	full_text
	
i64 %20
2add8B)
'
	full_text

%25 = add i64 %23, %24
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %24
4mul8B+
)
	full_text

%26 = mul nsw i64 %22, 5
%i648B

	full_text
	
i64 %22
2add8B)
'
	full_text

%27 = add i64 %25, %26
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %26
Ugetelementptr8BB
@
	full_text3
1
/%28 = getelementptr double, double* %0, i64 %27
%i648B

	full_text
	
i64 %27
@bitcast8B3
1
	full_text$
"
 %29 = bitcast double* %28 to i8*
-double*8B

	full_text

double* %28
ecall8B[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %29, i8 0, i64 40, i1 false)
%i8*8B

	full_text
	
i8* %29
'br8B

	full_text

br label %30
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
$i328B

	full_text


i32 %2
,double*8B

	full_text


double* %0
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
'i648B

	full_text

	i64 21125
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 40
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 5
%i648B

	full_text
	
i64 325
#i328B

	full_text	

i32 0
!i88B

	full_text

i8 0
#i328B

	full_text	

i32 2       	  
 

                     !    "# "" $% $$ &' && () (* (( +, ++ -. -/ -- 01 00 23 22 45 44 68 9 : ; 0   	 
            !  # % '$ )& *" ,( .+ /- 10 32 5  76 7 << == 7 <<  <<  << 4 == 4> > > > >  > "? $@ A 4B 4C +D &E F 4G "

exact_rhs1"
_Z13get_global_idj"
llvm.memset.p0i8.i64*?
npb-BT-exact_rhs1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes	
????

wgsize_log1p
???A

wgsize
@
 
transfer_bytes_log1p
???A

devmap_label
