

[external]
9allocaB/
-
	full_text 

%11 = alloca double, align 8
9allocaB/
-
	full_text 

%12 = alloca double, align 8
LcallBD
B
	full_text5
3
1%13 = tail call i64 @_Z13get_global_idj(i32 0) #4
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
>bitcastB3
1
	full_text$
"
 %15 = bitcast double* %11 to i8*
+double*B

	full_text

double* %11
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %15) #5
#i8*B

	full_text
	
i8* %15
>bitcastB3
1
	full_text$
"
 %16 = bitcast double* %12 to i8*
+double*B

	full_text

double* %12
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %16) #5
#i8*B

	full_text
	
i8* %16
hcallB`
^
	full_textQ
O
M%17 = tail call double @_Z3powdd(double 5.000000e+00, double 1.300000e+01) #4
5icmpB-
+
	full_text

%18 = icmp slt i32 %14, %7
#i32B

	full_text
	
i32 %14
4icmpB,
*
	full_text

%19 = icmp sgt i32 %14, 0
#i32B

	full_text
	
i32 %14
/andB(
&
	full_text

%20 = and i1 %18, %19
!i1B

	full_text


i1 %18
!i1B

	full_text


i1 %19
8brB2
0
	full_text#
!
br i1 %20, label %21, label %53
!i1B

	full_text


i1 %20
1shl8B(
&
	full_text

%22 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
^getelementptr8BK
I
	full_text<
:
8%24 = getelementptr inbounds double, double* %1, i64 %23
%i648B

	full_text
	
i64 %23
Abitcast8B4
2
	full_text%
#
!%25 = bitcast double* %24 to i64*
-double*8B

	full_text

double* %24
Hload8B>
<
	full_text/
-
+%26 = load i64, i64* %25, align 8, !tbaa !8
'i64*8B

	full_text


i64* %25
Abitcast8B4
2
	full_text%
#
!%27 = bitcast double* %12 to i64*
-double*8B

	full_text

double* %12
Hstore8B=
;
	full_text.
,
*store i64 %26, i64* %27, align 8, !tbaa !8
%i648B

	full_text
	
i64 %26
'i64*8B

	full_text


i64* %27
5icmp8B+
)
	full_text

%28 = icmp sgt i32 %6, 1
:br8B2
0
	full_text#
!
br i1 %28, label %29, label %53
#i18B

	full_text


i1 %28
Abitcast8B4
2
	full_text%
#
!%30 = bitcast double* %11 to i64*
-double*8B

	full_text

double* %11
5mul8B,
*
	full_text

%31 = mul nsw i32 %14, %3
%i328B

	full_text
	
i32 %14
5zext8B+
)
	full_text

%32 = zext i32 %6 to i64
Hstore8B=
;
	full_text.
,
*store i64 %26, i64* %30, align 8, !tbaa !8
%i648B

	full_text
	
i64 %26
'i64*8B

	full_text


i64* %30
0add8B'
%
	full_text

%33 = add i32 %31, 1
%i328B

	full_text
	
i32 %31
1mul8B(
&
	full_text

%34 = mul i32 %33, %2
%i328B

	full_text
	
i32 %33
4add8B+
)
	full_text

%35 = add nsw i32 %34, 1
%i328B

	full_text
	
i32 %34
6sext8B,
*
	full_text

%36 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
^getelementptr8BK
I
	full_text<
:
8%37 = getelementptr inbounds double, double* %0, i64 %36
%i648B

	full_text
	
i64 %36
gcall8B]
[
	full_textN
L
Jcall void @vranlc(i32 %8, double* nonnull %11, double %17, double* %37) #5
-double*8B

	full_text

double* %11
+double8B

	full_text


double %17
-double*8B

	full_text

double* %37
Ycall8BO
M
	full_text@
>
<%38 = call double @randlc(double* nonnull %12, double %9) #5
-double*8B

	full_text

double* %12
4icmp8B*
(
	full_text

%39 = icmp eq i32 %6, 2
:br8B2
0
	full_text#
!
br i1 %39, label %53, label %40
#i18B

	full_text


i1 %39
'br8B

	full_text

br label %41
Bphi8B9
7
	full_text*
(
&%42 = phi i64 [ %51, %41 ], [ 2, %40 ]
%i648B

	full_text
	
i64 %51
Hload8B>
<
	full_text/
-
+%43 = load i64, i64* %27, align 8, !tbaa !8
'i64*8B

	full_text


i64* %27
Hstore8B=
;
	full_text.
,
*store i64 %43, i64* %30, align 8, !tbaa !8
%i648B

	full_text
	
i64 %43
'i64*8B

	full_text


i64* %30
8trunc8B-
+
	full_text

%44 = trunc i64 %42 to i32
%i648B

	full_text
	
i64 %42
2add8B)
'
	full_text

%45 = add i32 %31, %44
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %44
1mul8B(
&
	full_text

%46 = mul i32 %45, %2
%i328B

	full_text
	
i32 %45
4add8B+
)
	full_text

%47 = add nsw i32 %46, 1
%i328B

	full_text
	
i32 %46
6sext8B,
*
	full_text

%48 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
^getelementptr8BK
I
	full_text<
:
8%49 = getelementptr inbounds double, double* %0, i64 %48
%i648B

	full_text
	
i64 %48
gcall8B]
[
	full_textN
L
Jcall void @vranlc(i32 %8, double* nonnull %11, double %17, double* %49) #5
-double*8B

	full_text

double* %11
+double8B

	full_text


double %17
-double*8B

	full_text

double* %49
Ycall8BO
M
	full_text@
>
<%50 = call double @randlc(double* nonnull %12, double %9) #5
-double*8B

	full_text

double* %12
8add8B/
-
	full_text 

%51 = add nuw nsw i64 %42, 1
%i648B

	full_text
	
i64 %42
7icmp8B-
+
	full_text

%52 = icmp eq i64 %51, %32
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %32
:br8B2
0
	full_text#
!
br i1 %52, label %53, label %41
#i18B

	full_text


i1 %52
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %16) #5
%i8*8B

	full_text
	
i8* %16
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %15) #5
%i8*8B

	full_text
	
i8* %15
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %1
*double8B

	full_text

	double %9
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %2
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 1.300000e+01
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2
4double8B&
$
	full_text

double 5.000000e+00
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 8
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0       	  
 

                     !    "# "" $% $& $$ '' () (+ ** ,- ,, .. /0 /1 // 23 22 45 44 67 66 89 88 :; :: <= <> <? << @A @@ BB CD CG FF HI HH JK JL JJ MN MM OP OQ OO RS RR TU TT VW VV XY XX Z[ Z\ Z] ZZ ^_ ^^ `a `` bc bd bb ef eh gg ij ii kl 'l .l Bm ,n o p @p ^q <q Zr :r Xs 4s R   	 
           ! #  %" &' ) + -  0* 1, 32 54 76 98 ; = >: ? AB D` G" IH K* LF N, PM QO SR UT WV Y [ \X ] _F a` c. db f
 h j  g( *( gC gC EE Fe ge F yy uu tt vv k ww xxi yy ig yy g uu @ xx @< ww < tt  vv ^ xx ^ tt Z ww Zz `{ | | | '| 2| 6| T} B~  F? ? ? g? i	? 	? ? 	? "
kernel_zran3_1"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"

_Z3powdd"
vranlc"
randlc"
llvm.lifetime.end.p0i8*?
npb-MG-kernel_zran3_1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

devmap_label

 
transfer_bytes_log1p
???A

transfer_bytes	
????

wgsize_log1p
???A

wgsize
